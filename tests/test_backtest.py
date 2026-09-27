"""Backtest engine: look-ahead safety, weight drift, transaction costs, strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equilibrium.backtest.engine import (
    BacktestConfig,
    Rebalance,
    rebalance_dates,
    run_backtest,
)
from equilibrium.backtest.strategies import (
    BlackLittermanStrategy,
    EqualWeight,
    MarketCap,
    MaxSharpeHistorical,
    MinVariance,
)
from equilibrium.bl.views import View, ViewType
from equilibrium.core.errors import InsufficientHistoryError
from equilibrium.core.types import Frequency
from equilibrium.optimize.constraints import Constraints
from tests.conftest import make_prices


def _cfg(**kw):
    base = dict(lookback=12, transaction_cost_bps=0.0, risk_free_rate=0.0)
    base.update(kw)
    return BacktestConfig(**base)


def test_rebalance_dates_are_last_trading_day():
    idx = pd.bdate_range("2024-01-01", "2024-06-30")
    m = rebalance_dates(idx, Rebalance.MONTHLY)
    assert len(m) == 6 and m[0] == pd.Timestamp("2024-01-31")
    assert m[1] == pd.Timestamp("2024-02-29")
    q = rebalance_dates(idx, Rebalance.QUARTERLY)
    assert list(q) == [pd.Timestamp("2024-03-29"), pd.Timestamp("2024-06-28")]


def test_equal_weight_without_costs_matches_manual_simulation():
    prices = make_prices(n_days=600)
    res = run_backtest(prices, EqualWeight(), _cfg())
    # independent re-implementation: monthly rebalanced equal weight with drift
    daily = prices.pct_change().dropna()
    targets = set(res.weights.index)
    w = None
    manual = []
    for day, r in daily.loc[daily.index >= res.weights.index[0]].iterrows():
        if w is None:
            manual.append(0.0)
        else:
            g = float(w @ r.to_numpy())
            w = w * (1 + r.to_numpy()) / (1 + g)
            manual.append(g)
        if day in targets:
            w = np.full(4, 0.25)
    np.testing.assert_allclose(res.returns.to_numpy(), manual, atol=1e-14)


def test_no_lookahead_future_prices_do_not_change_past_weights():
    prices = make_prices(n_days=700)
    base = run_backtest(prices, MinVariance(Constraints()), _cfg())
    cut = base.weights.index[5]
    shocked = prices.copy()
    after = shocked.index > cut
    shocked.loc[after, "AAA"] *= np.linspace(1, 5, after.sum())  # rewrite the future
    alt = run_backtest(shocked, MinVariance(Constraints()), _cfg())
    pd.testing.assert_frame_equal(base.weights.loc[:cut], alt.weights.loc[:cut])


def test_strategy_only_sees_data_up_to_rebalance_date():
    seen = []

    class Spy(EqualWeight):
        def weights(self, ctx):
            seen.append((ctx.date, ctx.prices.index[-1], ctx.returns.index[-1]))
            return super().weights(ctx)

    run_backtest(make_prices(n_days=500), Spy(), _cfg())
    for date, last_price, last_ret in seen:
        assert last_price == date and last_ret <= date


def test_transaction_costs_reduce_returns_by_expected_amount():
    prices = make_prices(n_days=600)
    free = run_backtest(prices, EqualWeight(), _cfg())
    costly = run_backtest(prices, EqualWeight(), _cfg(transaction_cost_bps=25.0))
    # initial purchase from cash: traded notional 1.0 -> cost 25 bps
    assert costly.costs.iloc[0] == pytest.approx(0.0025)
    assert costly.turnover.iloc[0] == pytest.approx(0.5)
    for day, cost in costly.costs.items():
        g = free.returns[day]
        assert costly.returns[day] == pytest.approx((1 + g) * (1 - cost) - 1, abs=1e-14)
    assert costly.summary.total_return < free.summary.total_return
    np.testing.assert_allclose(costly.costs.to_numpy(), 0.0025 * 2 * costly.turnover.to_numpy())


def test_all_strategies_run_and_report():
    prices = make_prices(n_days=900)
    caps = {t: 1e9 * (i + 1) for i, t in enumerate(prices.columns)}
    now = prices.iloc[-1]
    view = View(type=ViewType.RELATIVE, long="DDD", short="AAA", value=0.03, confidence=0.6)
    strategies = [
        EqualWeight(), MarketCap(caps, now), MinVariance(Constraints()),
        MaxSharpeHistorical(Constraints()),
        BlackLittermanStrategy(caps, now, Constraints(), 2.5, 0.05, [view]),
    ]
    for s in strategies:
        res = run_backtest(prices, s, _cfg(transaction_cost_bps=10))
        assert res.n_rebalances > 10
        np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-8)
        assert np.isfinite(res.summary.sharpe)
        assert res.summary.max_drawdown <= 0
    assert any("look-ahead" in w for w in strategies[1].warnings)
    assert any("hindsight" in w for w in strategies[4].warnings)


def test_market_cap_strategy_tracks_price_changes():
    prices = make_prices(n_days=400)
    caps = {t: 1.0 for t in prices.columns}
    res = run_backtest(prices, MarketCap(caps, prices.iloc[-1]), _cfg())
    last = res.weights.iloc[-1]
    p_t = prices.loc[res.weights.index[-1]]
    implied = (p_t / prices.iloc[-1]) / (p_t / prices.iloc[-1]).sum()
    np.testing.assert_allclose(last.to_numpy(), implied.to_numpy())


def test_quarterly_has_fewer_rebalances():
    prices = make_prices(n_days=900)
    m = run_backtest(prices, EqualWeight(), _cfg())
    q = run_backtest(prices, EqualWeight(), _cfg(rebalance=Rebalance.QUARTERLY))
    assert q.n_rebalances < m.n_rebalances


def test_insufficient_history_for_lookback():
    with pytest.raises(InsufficientHistoryError):
        run_backtest(make_prices(n_days=200), EqualWeight(), _cfg(lookback=36))


def test_weekly_estimation_frequency():
    res = run_backtest(make_prices(n_days=500), MinVariance(Constraints()),
                       _cfg(estimation_frequency=Frequency.WEEKLY, lookback=52))
    assert res.n_rebalances > 5
