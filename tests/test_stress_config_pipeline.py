"""Stress tests, configuration precedence, session state and the end-to-end pipeline."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from equilibrium.bl.views import View, ViewType
from equilibrium.config.session import Session, load_session, save_session
from equilibrium.config.settings import load_settings
from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import Weights
from equilibrium.data.synthetic import SyntheticProvider
from equilibrium.optimize.optimizer import Objective
from equilibrium.pipeline import (
    DataWindow,
    build_portfolio,
    load_snapshot,
    parse_weight_spec,
    run_black_litterman,
)
from equilibrium.stress.scenarios import (
    Scenario,
    historical_stress,
    hypothetical_stress,
    parse_shocks,
)


# ------------------------------------------------------------------ stress
def test_historical_stress_buy_and_hold():
    idx = pd.bdate_range("2020-02-17", "2020-03-27")
    a = np.linspace(100, 50, len(idx))
    b = np.linspace(100, 90, len(idx))
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    sc = Scenario("t", "Test", dt.date(2020, 2, 19), dt.date(2020, 3, 23), "")
    res = historical_stress(prices, Weights(("A", "B"), np.array([0.5, 0.5])), sc)
    win = prices.loc["2020-02-19":"2020-03-23"]
    ra = win["A"].iloc[-1] / win["A"].iloc[0] - 1
    rb = win["B"].iloc[-1] / win["B"].iloc[0] - 1
    assert res.portfolio_return == pytest.approx(0.5 * ra + 0.5 * rb)
    assert res.max_drawdown == pytest.approx(res.portfolio_return)  # monotone decline


def test_historical_stress_refuses_missing_history():
    idx = pd.bdate_range("2020-03-10", "2020-03-31")
    prices = pd.DataFrame({"A": np.linspace(1, 2, len(idx))}, index=idx)
    sc = Scenario("t", "Test", dt.date(2020, 2, 19), dt.date(2020, 3, 23), "")
    with pytest.raises(InsufficientHistoryError, match="no proxy"):
        historical_stress(prices, Weights(("A",), np.array([1.0])), sc)


def test_hypothetical_shock_direct_and_propagated(cov3):
    w = Weights(cov3.tickers, np.array([0.5, 0.3, 0.2]))
    res = hypothetical_stress(w, cov3, {"CCC": -0.30})
    assert res.direct_impact == pytest.approx(0.2 * -0.30)
    # E[r_A | r_C = s] = cov_AC / var_C * s
    assert res.propagated_asset_moves["AAA"] == pytest.approx(0.006 / 0.0225 * -0.30)
    assert res.propagated_asset_moves["BBB"] == pytest.approx(0.009 / 0.0225 * -0.30)
    expected = 0.5 * 0.006 / 0.0225 * -0.3 + 0.3 * 0.009 / 0.0225 * -0.3 + 0.2 * -0.3
    assert res.propagated_impact == pytest.approx(expected)


def test_parse_shocks():
    assert parse_shocks(["nvda=-0.3", "AAPL=-10%"]) == {"NVDA": -0.3, "AAPL": -0.10}
    with pytest.raises(ValidationError):
        parse_shocks(["NVDA:-0.3"])
    with pytest.raises(ValidationError):
        parse_shocks(["NVDA=__import__('os')"])
    with pytest.raises(ValidationError):
        parse_shocks(["NVDA=-2"])


# ------------------------------------------------------------------ config & session
def test_config_file_and_cli_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "equilibrium.yaml").write_text(
        "risk_free_rate: 0.03\ntau: 0.1\nconstraints:\n  max_weight: 0.4\n")
    s, found = load_settings()
    assert found is not None and s.risk_free_rate == 0.03 and s.tau == 0.1
    assert s.constraints.max_weight == 0.4
    s2, _ = load_settings(overrides={"risk_free_rate": 0.05, "max_weight": 0.5, "tau": None})
    assert s2.risk_free_rate == 0.05 and s2.tau == 0.1 and s2.constraints.max_weight == 0.5
    assert "risk_free_rate" in s2.model_fields_set and "risk_aversion" not in s2.model_fields_set


def test_invalid_config_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "equilibrium.yaml").write_text("tau: -1\n")
    with pytest.raises(ValidationError, match="tau"):
        load_settings()
    (tmp_path / "equilibrium.yaml").write_text("unknown_key: 1\n")
    with pytest.raises(ValidationError):
        load_settings()
    (tmp_path / "equilibrium.yaml").write_text("!!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(ValidationError):  # safe_load refuses to construct Python objects
        load_settings()


def test_session_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    s = Session(universe=["AAA", "BBB"],
                views=[View(type=ViewType.ABSOLUTE, asset="AAA", value=0.1, confidence=0.5)])
    save_session(s, path)
    assert load_session(path) == s


def test_parse_weight_spec():
    assert parse_weight_spec("spy=0.6, TLT=0.4") == {"SPY": 0.6, "TLT": 0.4}
    with pytest.raises(ValidationError):
        parse_weight_spec("SPY=abc")


# ------------------------------------------------------------------ pipeline end-to-end
@pytest.fixture
def snap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings, _ = load_settings(overrides={"risk_free_rate": 0.02})
    return load_snapshot(SyntheticProvider(), ["AAA", "BBB", "CCC", "DDD"], settings,
                         DataWindow(period="10y"))


def test_pipeline_no_views_gives_market_portfolio(snap):
    bl = run_black_litterman(snap, SyntheticProvider(), [])
    report = build_portfolio(snap, bl, Objective.MAX_SHARPE)
    np.testing.assert_allclose(report.optimized.weights.values, bl.market_weights.values,
                               atol=1e-5)
    assert any("No views" in a for a in bl.assumptions)


def test_pipeline_view_tilts_toward_view(snap):
    base = run_black_litterman(snap, SyntheticProvider(), [])
    view = View(type=ViewType.RELATIVE, long="DDD", short="AAA", value=0.10, confidence=0.8)
    bl = run_black_litterman(snap, SyntheticProvider(), [view])
    i_d, i_a = snap.tickers.index("DDD"), snap.tickers.index("AAA")
    spread_post = bl.posterior.posterior.values[i_d] - bl.posterior.posterior.values[i_a]
    spread_prior = base.posterior.posterior.values[i_d] - base.posterior.posterior.values[i_a]
    assert spread_prior < spread_post < 0.10
    report = build_portfolio(snap, bl, Objective.MAX_SHARPE)
    assert report.optimized.weights["DDD"] > base.market_weights["DDD"]
    assert report.optimized.weights["AAA"] < base.market_weights["AAA"]
    assert report.risk_contributions.sum() == pytest.approx(1.0)


def test_pipeline_explicit_market_weights(snap):
    bl = run_black_litterman(snap, SyntheticProvider(), [],
                             {"AAA": 4, "BBB": 3, "CCC": 2, "DDD": 1})
    np.testing.assert_allclose(bl.market_weights.values, [0.4, 0.3, 0.2, 0.1])
    with pytest.raises(ValidationError):
        run_black_litterman(snap, SyntheticProvider(), [], {"AAA": 1.0})


def test_pipeline_insufficient_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings, _ = load_settings()
    with pytest.raises(InsufficientHistoryError):
        load_snapshot(SyntheticProvider(), ["AAA", "BBB", "CCC", "DDD"], settings,
                      DataWindow(period="6mo"))


def test_views_outside_universe_are_set_aside_and_reported(snap):
    inside = View(type=ViewType.ABSOLUTE, asset="AAA", value=0.10)
    outside = View(type=ViewType.RELATIVE, long="ZZZ", short="AAA", value=0.05)
    bl = run_black_litterman(snap, SyntheticProvider(), [inside, outside])
    assert bl.views == [inside]
    assert bl.inactive_views == [outside]
    assert bl.posterior.n_views == 1
