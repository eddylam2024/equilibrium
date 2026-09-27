"""Application layer: orchestrates data -> risk -> equilibrium -> views -> optimisation.

Pure orchestration with no presentation code; the CLI, exporters and tests all call these
functions and render the returned dataclasses however they like.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from equilibrium.bl.equilibrium import equilibrium_returns, market_cap_weights
from equilibrium.bl.omega import OmegaMethod, build_omega, prior_view_variance
from equilibrium.bl.posterior import PosteriorResult, black_litterman
from equilibrium.bl.views import View, build_view_matrices
from equilibrium.config.settings import Settings
from equilibrium.core.errors import ValidationError
from equilibrium.core.types import (
    CovarianceMatrix,
    ExpectedReturns,
    ReturnKind,
    Source,
    Weights,
    normalize_tickers,
)
from equilibrium.data.provider import MarketDataProvider, PriceData, PriceRequest
from equilibrium.optimize.optimizer import Objective, OptimizationResult, optimize
from equilibrium.returns.calc import ReturnMethod, periodic_returns
from equilibrium.risk.covariance import correlation_from_covariance, estimate_covariance
from equilibrium.risk.metrics import (
    PerformanceSummary,
    percent_risk_contribution,
    performance_summary,
    portfolio_volatility,
)

INTERVAL_RANK = {"1d": 0, "1wk": 1, "1mo": 2}
FREQ_RANK = {"daily": 0, "weekly": 1, "monthly": 2}


@dataclass(frozen=True)
class DataWindow:
    period: str | None = None
    start: dt.date | None = None
    end: dt.date | None = None
    interval: str = "1d"


@dataclass
class MarketSnapshot:
    """Historical inputs: cleaned prices, periodic returns, and the estimated covariance."""

    tickers: tuple[str, ...]
    price_data: PriceData
    returns: pd.DataFrame
    cov: CovarianceMatrix
    corr: np.ndarray
    settings: Settings

    @property
    def start(self) -> dt.date:
        return self.price_data.prices.index[0].date()

    @property
    def end(self) -> dt.date:
        return self.price_data.prices.index[-1].date()

    @property
    def years(self) -> float:
        return (self.end - self.start).days / 365.25

    @property
    def synthetic(self) -> bool:
        return self.price_data.synthetic


def load_snapshot(provider: MarketDataProvider, tickers: Sequence[str], settings: Settings,
                  window: DataWindow, method: ReturnMethod = ReturnMethod.SIMPLE
                  ) -> MarketSnapshot:
    syms = normalize_tickers(tickers)
    if INTERVAL_RANK[window.interval] > FREQ_RANK[settings.frequency.value]:
        raise ValidationError(
            f"Interval {window.interval} is coarser than the {settings.frequency.value} return "
            "frequency; use a finer --interval or a lower --frequency."
        )
    request = PriceRequest(syms, start=window.start, end=window.end,
                           period=window.period or (None if window.start else
                                                    settings.lookback_period),
                           interval=window.interval)
    price_data = provider.get_prices(request)
    returns = periodic_returns(price_data.prices, settings.frequency, method)
    min_obs = len(syms) + 2
    if len(returns) < min_obs:
        from equilibrium.core.errors import InsufficientHistoryError

        raise InsufficientHistoryError(
            f"Only {len(returns)} {settings.frequency.value} returns for {len(syms)} assets; "
            f"need at least {min_obs} for a non-singular covariance. Use a longer --period, "
            "a higher --frequency, or --estimator ledoit-wolf."
        )
    cov = estimate_covariance(returns, settings.frequency.periods_per_year,
                              settings.covariance_estimator, settings.psd_policy)
    return MarketSnapshot(syms, price_data, returns, cov, correlation_from_covariance(cov.values),
                          settings)


# ------------------------------------------------------------------ analysis
@dataclass
class AssetStatistics:
    ticker: str
    summary: PerformanceSummary
    mean_return: float  # annualised arithmetic mean, historical


def asset_statistics(snap: MarketSnapshot) -> list[AssetStatistics]:
    ppy = snap.settings.frequency.periods_per_year
    rf = snap.settings.risk_free_rate
    return [AssetStatistics(t, performance_summary(snap.returns[t], rf, ppy),
                            float(snap.returns[t].mean() * ppy)) for t in snap.tickers]


def historical_mean_returns(snap: MarketSnapshot) -> ExpectedReturns:
    """Sample-mean excess returns: a description of the past, not a forecast."""
    ppy = snap.settings.frequency.periods_per_year
    mean = snap.returns.mean().to_numpy() * ppy - snap.settings.risk_free_rate
    return ExpectedReturns(snap.tickers, mean, ReturnKind.EXCESS, Source.HISTORICAL,
                           "historical sample mean")


# ------------------------------------------------------------------ Black-Litterman
@dataclass
class BLRun:
    market_weights: Weights
    market_weights_source: str
    posterior: PosteriorResult
    views: list[View]
    omega_methods: list[OmegaMethod]
    prior_view_variance: np.ndarray
    assumptions: list[str] = field(default_factory=list)
    inactive_views: list[View] = field(default_factory=list)


def split_views(views: Sequence[View], tickers: Sequence[str]) -> tuple[list[View], list[View]]:
    """Views whose assets are all in the universe (active), and the rest (inactive).

    Saved views persist across universes; a view on an asset outside the current universe
    cannot be expressed in P, so it is set aside and reported, never silently dropped.
    """
    universe = set(tickers)
    active = [v for v in views if set(v.assets) <= universe]
    return active, [v for v in views if not set(v.assets) <= universe]


def resolve_market_weights(provider: MarketDataProvider, tickers: Sequence[str],
                           explicit: dict[str, float] | None) -> tuple[Weights, str]:
    if explicit:
        missing = [t for t in tickers if t not in explicit]
        extra = [t for t in explicit if t not in tickers]
        if missing or extra:
            raise ValidationError(
                "Market weights must cover exactly the universe. "
                + (f"Missing: {', '.join(missing)}. " if missing else "")
                + (f"Not in universe: {', '.join(extra)}." if extra else ""))
        return (Weights.normalized(tickers, [explicit[t] for t in tickers]),
                "user-supplied (normalised to 100%)")
    caps = provider.get_market_caps(tickers)
    src = "synthetic caps" if provider.synthetic else f"current market caps ({provider.name})"
    return market_cap_weights(caps, tickers), src


def run_black_litterman(snap: MarketSnapshot, provider: MarketDataProvider, views: list[View],
                        market_weights: dict[str, float] | None = None) -> BLRun:
    s = snap.settings
    views, inactive = split_views(views, snap.tickers)
    w_mkt, w_src = resolve_market_weights(provider, snap.tickers, market_weights)
    pi = equilibrium_returns(snap.cov, w_mkt, s.risk_aversion)
    P, Q = build_view_matrices(views, snap.tickers, s.risk_free_rate)
    omega, methods = build_omega(views, P, snap.cov.values, s.tau)
    post = black_litterman(pi, snap.cov, P, Q, omega, s.tau)
    return BLRun(w_mkt, w_src, post, list(views), methods,
                 prior_view_variance(P, snap.cov.values, s.tau),
                 assumptions=describe_assumptions(snap, w_src, methods), inactive_views=inactive)


def describe_assumptions(snap: MarketSnapshot, w_src: str, methods: list[OmegaMethod]
                         ) -> list[str]:
    s = snap.settings
    set_fields = s.model_fields_set

    def origin(name: str, default_note: str) -> str:
        return "set" if name in set_fields else f"default — {default_note}"

    out = [
        f"r_f = {s.risk_free_rate:.2%} ({origin('risk_free_rate', 'override with --rf')})",
        f"δ = {s.risk_aversion:g} ({origin('risk_aversion', 'He & Litterman 1999')})",
        f"τ = {s.tau:g} ({origin('tau', 'He & Litterman 1999')})",
        f"Market portfolio = the {len(snap.tickers)} selected assets, weighted by {w_src}; "
        "not the global market.",
        f"Σ = {s.covariance_estimator.value} covariance of {s.frequency.value} simple returns, "
        f"annualised ×{s.frequency.periods_per_year}.",
    ]
    if methods:
        used = sorted({m.value for m in methods})
        out.append("Ω diagonal (independent view errors); per-view method: " + ", ".join(used)
                   + ".")
    else:
        out.append("No views: posterior mean equals equilibrium Π exactly.")
    return out


# ------------------------------------------------------------------ portfolio construction
@dataclass
class PortfolioReport:
    snapshot: MarketSnapshot
    bl: BLRun
    optimized: OptimizationResult
    market_expected_excess: float
    market_volatility: float
    risk_contributions: np.ndarray

    @property
    def rf(self) -> float:
        return self.snapshot.settings.risk_free_rate


def build_portfolio(snap: MarketSnapshot, bl: BLRun, objective: Objective = Objective.MAX_SHARPE,
                    target_total_return: float | None = None) -> PortfolioReport:
    """Optimise on the BL posterior mean and posterior predictive covariance."""
    s = snap.settings
    cov_post = bl.posterior.predictive_cov
    mu = bl.posterior.posterior
    target = None if target_total_return is None else target_total_return - s.risk_free_rate
    res = optimize(objective, cov_post, mu, s.constraints, target=target,
                   risk_aversion=s.risk_aversion)
    w_mkt = bl.market_weights.aligned_to(snap.tickers)
    return PortfolioReport(
        snapshot=snap, bl=bl, optimized=res,
        market_expected_excess=float(w_mkt @ mu.values),
        market_volatility=portfolio_volatility(w_mkt, cov_post.values),
        risk_contributions=percent_risk_contribution(res.weights.values, cov_post.values),
    )


def parse_weight_spec(spec: str | None) -> dict[str, float] | None:
    """Parse ``"AAPL=0.6,MSFT=0.4"`` (numbers only; nothing is evaluated)."""
    if not spec:
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValidationError(f"Weight {part!r} must look like TICKER=0.25")
        k, v = part.split("=", 1)
        try:
            out[k.strip().upper()] = float(v)
        except ValueError as exc:
            raise ValidationError(f"Weight {v!r} is not a number.") from exc
    return out
