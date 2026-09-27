"""Allocation methodologies evaluated by the backtester.

Each strategy receives a ``RebalanceContext`` built from information available at the close
of the rebalance date only. They are alternative methodologies, not a ranking.

Market-capitalisation weights in a backtest
-------------------------------------------
Free data sources (including Yahoo Finance) do not provide historical market caps. The
``MarketCap`` and ``BlackLitterman`` strategies therefore approximate the cap of asset i at
date t as  cap_i,now * P_i,t / P_i,now,  i.e. *today's* share count times the historical
price. This uses one piece of future information (the current share count, which reflects
subsequent buybacks and issuance) and is disclosed as a look-ahead warning on every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from equilibrium.bl.equilibrium import equilibrium_returns
from equilibrium.bl.omega import build_omega
from equilibrium.bl.posterior import black_litterman
from equilibrium.bl.views import View, build_view_matrices
from equilibrium.core.errors import InfeasibleError
from equilibrium.core.types import CovarianceMatrix, ExpectedReturns, ReturnKind, Source, Weights
from equilibrium.optimize.constraints import Constraints
from equilibrium.optimize.optimizer import max_sharpe, min_variance


@dataclass
class RebalanceContext:
    date: pd.Timestamp
    tickers: tuple[str, ...]
    prices: pd.DataFrame  # daily prices up to and including ``date``
    returns: pd.DataFrame  # estimation-frequency returns, trailing lookback window
    cov: CovarianceMatrix  # annualised, from ``returns``
    periods_per_year: int
    risk_free_rate: float


class Strategy(Protocol):
    name: str
    warnings: list[str]
    fallbacks: int

    def weights(self, ctx: RebalanceContext) -> np.ndarray: ...


@dataclass
class EqualWeight:
    name: str = "Equal weight"
    warnings: list[str] = field(default_factory=list)
    fallbacks: int = 0

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        return np.full(len(ctx.tickers), 1.0 / len(ctx.tickers))


def _approx_cap_weights(ctx: RebalanceContext, caps_now: dict[str, float],
                        prices_now: pd.Series) -> np.ndarray:
    p_t = ctx.prices.iloc[-1]
    caps = np.array([caps_now[t] * p_t[t] / prices_now[t] for t in ctx.tickers])
    return caps / caps.sum()


CAP_LOOKAHEAD_WARNING = (
    "Market weights use today's share counts x historical prices (no historical market caps "
    "are available): this contains look-ahead information."
)


@dataclass
class MarketCap:
    caps_now: dict[str, float]
    prices_now: pd.Series
    name: str = "Market-cap weight"
    warnings: list[str] = field(default_factory=lambda: [CAP_LOOKAHEAD_WARNING])
    fallbacks: int = 0

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        return _approx_cap_weights(ctx, self.caps_now, self.prices_now)


@dataclass
class MinVariance:
    constraints: Constraints
    name: str = "Minimum variance"
    warnings: list[str] = field(default_factory=list)
    fallbacks: int = 0

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        return min_variance(ctx.cov, self.constraints).weights.values


def _sharpe_or_fallback(strategy: Strategy, cov: CovarianceMatrix, mu: ExpectedReturns,
                        constraints: Constraints) -> np.ndarray:
    try:
        return max_sharpe(cov, mu, constraints).weights.values
    except InfeasibleError:
        strategy.fallbacks += 1  # reported: no positive excess return -> min variance
        return min_variance(cov, constraints).weights.values


@dataclass
class MaxSharpeHistorical:
    constraints: Constraints
    name: str = "Max Sharpe (historical mean)"
    warnings: list[str] = field(default_factory=list)
    fallbacks: int = 0

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        mean = ctx.returns.mean().to_numpy() * ctx.periods_per_year - ctx.risk_free_rate
        mu = ExpectedReturns(ctx.tickers, mean, ReturnKind.EXCESS, Source.HISTORICAL)
        return _sharpe_or_fallback(self, ctx.cov, mu, self.constraints)


@dataclass
class BlackLittermanStrategy:
    caps_now: dict[str, float]
    prices_now: pd.Series
    constraints: Constraints
    risk_aversion: float
    tau: float
    views: list[View] = field(default_factory=list)
    name: str = "Black-Litterman"
    warnings: list[str] = field(default_factory=list)
    fallbacks: int = 0

    def __post_init__(self) -> None:
        self.warnings.append(CAP_LOOKAHEAD_WARNING)
        if self.views:
            self.warnings.append(
                "Today's investor views are applied at every historical date. Views formed "
                "with hindsight make this an illustration, not an out-of-sample test."
            )

    def weights(self, ctx: RebalanceContext) -> np.ndarray:
        w_mkt = Weights(ctx.tickers, _approx_cap_weights(ctx, self.caps_now, self.prices_now))
        pi = equilibrium_returns(ctx.cov, w_mkt, self.risk_aversion)
        P, Q = build_view_matrices(self.views, ctx.tickers, ctx.risk_free_rate)
        omega, _ = build_omega(self.views, P, ctx.cov.values, self.tau)
        post = black_litterman(pi, ctx.cov, P, Q, omega, self.tau)
        return _sharpe_or_fallback(self, post.predictive_cov, post.posterior, self.constraints)
