"""Walk-forward backtest without look-ahead.

Timeline at each rebalance date t (last trading day of the month/quarter):
  1. The strategy sees daily prices up to and including the close of t, and the trailing
     ``lookback`` estimation-frequency returns ending at t. Nothing after t is visible.
  2. Covariance is estimated from that window only.
  3. Target weights are computed and traded at the close of t.
  4. From t+1 the portfolio earns daily returns; weights drift with prices:
        w_i <- w_i (1 + R_i) / (1 + R_p)
  5. At the next rebalance the drifted weights are traded back to target.

Transaction costs: ``cost = c * sum_i |w_target,i - w_drift,i|`` with ``c`` in bps of traded
notional (buys and sells both pay). Reported turnover is one-way: ``0.5 * sum_i |dw_i|``.
The initial purchase from cash is included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from equilibrium.backtest.strategies import RebalanceContext, Strategy
from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import Frequency
from equilibrium.returns.calc import periodic_returns, simple_returns
from equilibrium.risk.covariance import CovarianceEstimator, estimate_covariance
from equilibrium.risk.metrics import PerformanceSummary, performance_summary
from equilibrium.risk.numerics import PSDPolicy

TRADING_DAYS = 252


class Rebalance(StrEnum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"

    @property
    def period(self) -> str:
        return "M" if self is Rebalance.MONTHLY else "Q"


@dataclass(frozen=True)
class BacktestConfig:
    rebalance: Rebalance = Rebalance.MONTHLY
    lookback: int = 36  # observations at ``estimation_frequency``
    estimation_frequency: Frequency = Frequency.MONTHLY
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.04
    estimator: CovarianceEstimator = CovarianceEstimator.SAMPLE
    psd_policy: PSDPolicy = PSDPolicy.REPAIR

    def __post_init__(self) -> None:
        if self.lookback < 3:
            raise ValidationError("lookback must be at least 3 observations.")
        if not 0 <= self.transaction_cost_bps < 1000:
            raise ValidationError("transaction cost must be in [0, 1000) bps.")


@dataclass
class BacktestResult:
    strategy: str
    returns: pd.Series  # daily net returns, historical simulation
    weights: pd.DataFrame  # target weights at each rebalance
    turnover: pd.Series  # one-way turnover at each rebalance
    costs: pd.Series  # cost as a fraction of NAV at each rebalance
    summary: PerformanceSummary
    warnings: list[str] = field(default_factory=list)
    fallbacks: int = 0

    @property
    def n_rebalances(self) -> int:
        return len(self.weights)

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.iloc[1:].mean()) if len(self.turnover) > 1 else 0.0

    @property
    def annual_turnover(self) -> float:
        years = len(self.returns) / TRADING_DAYS
        return float(self.turnover.iloc[1:].sum() / years) if years > 0 else 0.0


def rebalance_dates(index: pd.DatetimeIndex, rebalance: Rebalance) -> pd.DatetimeIndex:
    """Last available trading day in each calendar month/quarter."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.to_period(rebalance.period)).max().to_numpy())


def run_backtest(prices: pd.DataFrame, strategy: Strategy, config: BacktestConfig
                 ) -> BacktestResult:
    tickers = tuple(str(c) for c in prices.columns)
    daily = simple_returns(prices)
    ppy = config.estimation_frequency.periods_per_year
    cost_rate = config.transaction_cost_bps / 1e4

    candidates = rebalance_dates(pd.DatetimeIndex(prices.index), config.rebalance)
    candidates = candidates[candidates < prices.index[-1]]  # need at least one day after

    weights_log: dict[pd.Timestamp, np.ndarray] = {}
    turnover_log: dict[pd.Timestamp, float] = {}
    cost_log: dict[pd.Timestamp, float] = {}
    targets: dict[pd.Timestamp, np.ndarray] = {}
    for t in candidates:
        window_prices = prices.loc[:t]  # information set: closes up to and including t
        try:
            est = periodic_returns(window_prices, config.estimation_frequency)
        except InsufficientHistoryError:
            continue  # not enough history yet at this date; stay in cash until there is
        if len(est) < config.lookback:
            continue
        est = est.iloc[-config.lookback:]
        cov = estimate_covariance(est, ppy, config.estimator, config.psd_policy)
        ctx = RebalanceContext(t, tickers, window_prices, est, cov, ppy, config.risk_free_rate)
        w = np.asarray(strategy.weights(ctx), dtype=np.float64)
        if w.shape != (len(tickers),) or abs(w.sum() - 1) > 1e-6:
            raise ValidationError(f"{strategy.name} produced invalid weights at {t.date()}.")
        targets[t] = w
    if not targets:
        raise InsufficientHistoryError(
            f"No rebalance date has {config.lookback} {config.estimation_frequency.value} "
            "observations of history. Use a longer period or a shorter --lookback."
        )

    first = min(targets)
    sim = daily.loc[daily.index >= first]
    held = np.zeros(len(tickers))
    out = np.empty(len(sim))
    r = sim.to_numpy()
    for i, day in enumerate(sim.index):
        if day == first:
            gross = 0.0  # position is opened at this close; no return earned yet
        else:
            gross = float(held @ r[i])
            held = held * (1.0 + r[i]) / (1.0 + gross)
        if day in targets:
            target = targets[day]
            traded = float(np.abs(target - held).sum())
            cost = cost_rate * traded
            turnover_log[day] = 0.5 * traded
            cost_log[day] = cost
            weights_log[day] = target
            held = target.copy()
            out[i] = (1.0 + gross) * (1.0 - cost) - 1.0
        else:
            out[i] = gross
    returns = pd.Series(out, index=sim.index, name=strategy.name)
    return BacktestResult(
        strategy=strategy.name,
        returns=returns,
        weights=pd.DataFrame.from_dict(weights_log, orient="index", columns=list(tickers)),
        turnover=pd.Series(turnover_log, name="turnover"),
        costs=pd.Series(cost_log, name="cost"),
        summary=performance_summary(returns, config.risk_free_rate, TRADING_DAYS),
        warnings=list(strategy.warnings),
        fallbacks=strategy.fallbacks,
    )
