"""Stress testing.

Historical scenarios (observed data)
------------------------------------
The portfolio is held buy-and-hold from the scenario start to its end. With no rebalancing the
portfolio return is exactly ``sum_i w_i R_i`` where ``R_i`` is asset i's cumulative return
over the window. Every asset must have prices covering the whole window; otherwise the
scenario is refused (no proxies are substituted).

Hypothetical shocks (model estimates)
-------------------------------------
Given shocks ``s_K`` to a subset K of assets:
  * direct impact: ``sum_{i in K} w_i s_i`` (unshocked assets assumed flat).
  * covariance-propagated impact: unshocked assets move by their conditional expectation
    under a zero-mean joint normal model, ``E[r_U | r_K = s_K] = Sigma_UK Sigma_KK^-1 s_K``.
    This is a linear-Gaussian approximation; real crises exhibit higher correlations and
    fatter tails than a covariance matrix estimated in normal times.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import CovarianceMatrix, Weights
from equilibrium.risk.numerics import solve_spd


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str
    start: dt.date
    end: dt.date
    description: str


# Windows are peak-to-trough of the S&P 500 index, the conventional dating.
SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in (
        Scenario("dotcom", "Dot-com bust", dt.date(2000, 3, 24), dt.date(2002, 10, 9),
                 "S&P 500 peak (24 Mar 2000) to trough (9 Oct 2002)."),
        Scenario("gfc", "Global Financial Crisis", dt.date(2007, 10, 9), dt.date(2009, 3, 9),
                 "S&P 500 peak (9 Oct 2007) to trough (9 Mar 2009)."),
        Scenario("euro2011", "2011 US downgrade / euro crisis", dt.date(2011, 4, 29),
                 dt.date(2011, 10, 3), "S&P 500 peak (29 Apr 2011) to trough (3 Oct 2011)."),
        Scenario("covid", "COVID-19 crash", dt.date(2020, 2, 19), dt.date(2020, 3, 23),
                 "S&P 500 peak (19 Feb 2020) to trough (23 Mar 2020)."),
        Scenario("rates2022", "2022 rate shock", dt.date(2022, 1, 3), dt.date(2022, 10, 12),
                 "S&P 500 peak (3 Jan 2022) to trough (12 Oct 2022)."),
    )
}


@dataclass(frozen=True)
class HistoricalStressResult:
    scenario: Scenario
    asset_returns: pd.Series
    portfolio_return: float
    max_drawdown: float
    first_date: dt.date
    last_date: dt.date


@dataclass(frozen=True)
class HypotheticalStressResult:
    shocks: dict[str, float]
    direct_asset_moves: pd.Series
    propagated_asset_moves: pd.Series
    direct_impact: float
    propagated_impact: float


def historical_stress(prices: pd.DataFrame, weights: Weights, scenario: Scenario
                      ) -> HistoricalStressResult:
    """Buy-and-hold portfolio return over a historical window (prices must cover it)."""
    w = weights.aligned_to(list(prices.columns))
    window = prices.loc[pd.Timestamp(scenario.start):pd.Timestamp(scenario.end)]
    if window.empty:
        raise InsufficientHistoryError(f"No prices inside the {scenario.name} window.")
    first_valid = window.apply(lambda s: s.first_valid_index())
    tolerance = pd.Timedelta(days=7)
    late = [t for t in prices.columns
            if first_valid[t] is None or first_valid[t] - pd.Timestamp(scenario.start) > tolerance]
    if late:
        raise InsufficientHistoryError(
            f"{', '.join(late)} did not trade at the start of the {scenario.name} window "
            f"({scenario.start}). Remove them or choose another scenario; no proxy is used."
        )
    window = window.ffill().dropna()
    rel = window / window.iloc[0]
    asset_ret = rel.iloc[-1] - 1.0
    wealth = rel.to_numpy() @ w  # buy-and-hold: value = sum w_i P_i,t / P_i,0
    wealth_s = pd.Series(wealth, index=window.index)
    mdd = float((wealth_s / wealth_s.cummax() - 1.0).min())
    return HistoricalStressResult(scenario, asset_ret, float(wealth[-1] - 1.0), mdd,
                                  window.index[0].date(), window.index[-1].date())


def parse_shocks(specs: list[str]) -> dict[str, float]:
    """Parse ``["NVDA=-0.30", "AAPL=-10%"]`` into ``{"NVDA": -0.30, "AAPL": -0.10}``."""
    out: dict[str, float] = {}
    for spec in specs:
        for part in spec.split(","):
            if "=" not in part:
                raise ValidationError(f"Shock {part!r} must look like TICKER=-0.30")
            k, v = part.split("=", 1)
            v = v.strip()
            try:
                val = float(v[:-1]) / 100 if v.endswith("%") else float(v)
            except ValueError as exc:
                raise ValidationError(f"Shock value {v!r} is not a number.") from exc
            if not -1.0 <= val <= 10.0:
                raise ValidationError(f"Shock {val:+.2%} is outside [-100%, +1000%].")
            out[k.strip().upper()] = val
    if not out:
        raise ValidationError("Provide at least one shock, e.g. --shock NVDA=-0.30")
    return out


def hypothetical_stress(weights: Weights, cov: CovarianceMatrix, shocks: dict[str, float]
                        ) -> HypotheticalStressResult:
    tickers = list(cov.tickers)
    unknown = [k for k in shocks if k not in tickers]
    if unknown:
        raise ValidationError(f"Shocked assets not in the portfolio: {', '.join(unknown)}")
    w = weights.aligned_to(tickers)
    k_idx = [tickers.index(k) for k in shocks]
    u_idx = [i for i in range(len(tickers)) if i not in k_idx]
    s = np.array([shocks[tickers[i]] for i in k_idx])

    direct = np.zeros(len(tickers))
    direct[k_idx] = s
    prop = direct.copy()
    if u_idx:
        sig = cov.values
        beta = solve_spd(sig[np.ix_(k_idx, k_idx)], s, "Sigma_KK")
        prop[u_idx] = sig[np.ix_(u_idx, k_idx)] @ beta
    return HypotheticalStressResult(
        shocks=shocks,
        direct_asset_moves=pd.Series(direct, index=tickers),
        propagated_asset_moves=pd.Series(prop, index=tickers),
        direct_impact=float(w @ direct),
        propagated_impact=float(w @ prop),
    )
