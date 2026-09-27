"""Returns, covariance, correlation and risk metrics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from equilibrium.core.errors import (
    InsufficientHistoryError,
    NumericalError,
    ValidationError,
)
from equilibrium.core.types import Frequency, Weights, normalize_tickers
from equilibrium.returns.calc import (
    ReturnMethod,
    log_returns,
    periodic_rate,
    periodic_returns,
    simple_returns,
)
from equilibrium.risk.covariance import (
    CovarianceEstimator,
    correlation_from_covariance,
    estimate_covariance,
    ledoit_wolf_covariance,
    sample_covariance,
)
from equilibrium.risk.metrics import (
    cagr,
    component_risk_contribution,
    downside_deviation,
    marginal_risk_contribution,
    max_drawdown,
    portfolio_variance,
    portfolio_volatility,
    sharpe_ratio,
    sortino_ratio,
)
from equilibrium.risk.numerics import PSDPolicy, is_psd, nearest_psd, validate_covariance


# ------------------------------------------------------------------ returns
def test_simple_returns_hand_example():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0], "B": [50.0, 50.0, 55.0]},
                     index=pd.date_range("2024-01-01", periods=3))
    r = simple_returns(p)
    # 110/100-1 = 0.10 ; 99/110-1 = -0.10 ; 50/50-1 = 0 ; 55/50-1 = 0.10
    np.testing.assert_allclose(r["A"].to_numpy(), [0.10, -0.10])
    np.testing.assert_allclose(r["B"].to_numpy(), [0.0, 0.10])


def test_log_returns_hand_example():
    p = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.date_range("2024-01-01", periods=3))
    r = log_returns(p)
    np.testing.assert_allclose(r["A"].to_numpy(), [math.log(1.1), math.log(0.9)])
    # log returns add up through time: sum = ln(99/100)
    assert r["A"].sum() == pytest.approx(math.log(0.99))


def test_monthly_returns_use_period_end_prices():
    idx = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-10", "2024-02-29", "2024-03-29"])
    p = pd.DataFrame({"A": [90.0, 100.0, 130.0, 120.0, 132.0]}, index=idx)
    r = periodic_returns(p, Frequency.MONTHLY, ReturnMethod.SIMPLE)
    # month-ends: Jan 100, Feb 120, Mar 132 -> 20%, 10%
    np.testing.assert_allclose(r["A"].to_numpy(), [0.20, 0.10])


def test_weekly_returns(prices):
    r = periodic_returns(prices, Frequency.WEEKLY)
    assert len(r) == len(prices.resample("W-FRI").last()) - 1


def test_nonpositive_prices_rejected():
    p = pd.DataFrame({"A": [1.0, 0.0]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValidationError):
        simple_returns(p)


def test_insufficient_history():
    p = pd.DataFrame({"A": [1.0]}, index=pd.date_range("2024-01-31", periods=1))
    with pytest.raises(InsufficientHistoryError):
        periodic_returns(p, Frequency.MONTHLY)


def test_periodic_rate():
    assert periodic_rate(0.12, 12) == pytest.approx(1.12 ** (1 / 12) - 1)


def test_ticker_validation():
    assert normalize_tickers([" aapl", "brk-b"]) == ("AAPL", "BRK-B")
    with pytest.raises(ValidationError):
        normalize_tickers(["AAPL; rm -rf /"])
    with pytest.raises(ValidationError):
        normalize_tickers(["AAPL", "aapl"])


# ------------------------------------------------------------------ covariance / correlation
def test_sample_covariance_hand_example():
    r = pd.DataFrame({"A": [0.01, 0.03, -0.01], "B": [0.02, 0.00, 0.01]})
    # means: A=0.01, B=0.01 ; deviations A=(0, .02, -.02), B=(.01, -.01, 0)
    # var A = (0+.0004+.0004)/2 = .0004 ; var B = (.0001+.0001+0)/2 = .0001
    # cov AB = (0 + (-.0002) + 0)/2 = -.0001
    s = sample_covariance(r)
    np.testing.assert_allclose(s, [[0.0004, -0.0001], [-0.0001, 0.0001]], atol=1e-15)
    np.testing.assert_allclose(s, r.cov().to_numpy())


def test_annualised_covariance_and_correlation(prices):
    r = periodic_returns(prices, Frequency.MONTHLY)
    cov = estimate_covariance(r, 12)
    np.testing.assert_allclose(cov.values, r.cov().to_numpy() * 12)
    corr = correlation_from_covariance(cov.values)
    np.testing.assert_allclose(corr, r.corr().to_numpy(), atol=1e-12)
    np.testing.assert_allclose(np.diag(corr), 1.0)


def test_correlation_is_not_covariance(cov3):
    corr = correlation_from_covariance(cov3.values)
    assert corr[0, 1] == pytest.approx(0.2)
    assert cov3.values[0, 1] == pytest.approx(0.012)


def test_ledoit_wolf_matches_independent_implementation(prices):
    r = periodic_returns(prices, Frequency.WEEKLY)
    lw, shrink = ledoit_wolf_covariance(r)
    # independent vectorised re-derivation of the LW(2004) formulas
    x = r.to_numpy()
    t, n = x.shape
    xc = x - x.mean(0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    d2 = np.linalg.norm(s - mu * np.eye(n), "fro") ** 2 / n
    y = xc**2
    b_bar2 = (np.sum(y.T @ y) / t - np.sum(s**2)) / (n * t)
    delta = min(b_bar2, d2) / d2
    np.testing.assert_allclose(shrink, delta, rtol=1e-10)
    np.testing.assert_allclose(lw, delta * mu * np.eye(n) + (1 - delta) * s, rtol=1e-10)
    assert 0 <= shrink <= 1


def test_ledoit_wolf_estimator_is_well_conditioned():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.05, size=(8, 10)))  # T < N: sample cov singular
    sample = estimate_covariance(r, 12, CovarianceEstimator.SAMPLE)
    assert np.linalg.matrix_rank(sample.values) <= 7
    assert "singular" in str(sample.diagnostics["warning"])
    lw = estimate_covariance(r, 12, CovarianceEstimator.LEDOIT_WOLF)
    assert np.linalg.eigvalsh(lw.values)[0] > 0
    assert lw.diagnostics["condition_number"] < 1e6


def test_nan_returns_rejected():
    r = pd.DataFrame({"A": [0.1, np.nan, 0.2], "B": [0.1, 0.2, 0.3]})
    with pytest.raises(ValidationError):
        sample_covariance(r)


# ------------------------------------------------------------------ PSD / numerical safety
def test_non_psd_raises_or_repairs():
    bad = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, -0.9], [0.7, -0.9, 1.0]])
    assert not is_psd(bad)
    with pytest.raises(NumericalError):
        validate_covariance(bad, PSDPolicy.RAISE)
    fixed, diag = validate_covariance(bad, PSDPolicy.REPAIR)
    assert diag.repaired and not diag.is_psd
    assert is_psd(fixed)
    np.testing.assert_allclose(fixed, fixed.T)


def test_nearest_psd_leaves_psd_matrix_unchanged(cov3):
    np.testing.assert_allclose(nearest_psd(cov3.values), cov3.values, atol=1e-12)


def test_asymmetric_rejected():
    with pytest.raises(NumericalError):
        validate_covariance(np.array([[1.0, 0.5], [0.4, 1.0]]))


def test_singular_covariance_detected():
    from equilibrium.risk.numerics import solve_spd

    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(NumericalError):
        solve_spd(singular, np.ones(2))


# ------------------------------------------------------------------ portfolio risk
def test_portfolio_variance_hand_example(cov3):
    w = np.array([0.5, 0.3, 0.2])
    # w'Sw = .25*.04 + .09*.09 + .04*.0225 + 2(.5*.3*.012 + .5*.2*.006 + .3*.2*.009)
    expected = 0.01 + 0.0081 + 0.0009 + 2 * (0.0018 + 0.0006 + 0.00054)
    assert portfolio_variance(w, cov3.values) == pytest.approx(expected)
    assert portfolio_volatility(w, cov3.values) == pytest.approx(math.sqrt(expected))


def test_risk_contributions_sum_to_volatility(cov3):
    w = np.array([0.5, 0.3, 0.2])
    ccr = component_risk_contribution(w, cov3.values)
    assert ccr.sum() == pytest.approx(portfolio_volatility(w, cov3.values))
    mcr = marginal_risk_contribution(w, cov3.values)
    # finite-difference check of d sigma / d w_i
    eps = 1e-7
    for i in range(3):
        w2 = w.copy()
        w2[i] += eps
        fd = (portfolio_volatility(w2, cov3.values) - portfolio_volatility(w, cov3.values)) / eps
        assert mcr[i] == pytest.approx(fd, rel=1e-5)


def test_dimension_mismatch(cov3):
    with pytest.raises(ValidationError):
        portfolio_variance(np.array([0.5, 0.5]), cov3.values)


def test_weights_alignment_and_normalisation():
    w = Weights.normalized(["A", "B"], [3.0, 1.0])
    np.testing.assert_allclose(w.values, [0.75, 0.25])
    np.testing.assert_allclose(w.aligned_to(["B", "A"]), [0.25, 0.75])
    with pytest.raises(ValidationError):
        w.aligned_to(["A", "C"])


# ------------------------------------------------------------------ ex-post statistics
def test_sharpe_hand_example():
    r = pd.Series([0.02, 0.00, 0.04, -0.02])
    # rf = 0 -> mean .01, sample sd = sqrt(((.01)^2+(.01)^2+(.03)^2+(.03)^2)/3) = sqrt(.002/3)
    expected = 0.01 / math.sqrt(0.002 / 3) * math.sqrt(12)
    assert sharpe_ratio(r, 0.0, 12) == pytest.approx(expected)


def test_downside_deviation_and_sortino():
    r = pd.Series([0.02, -0.01, 0.03, -0.03])
    # shortfalls below 0: (0, -.01, 0, -.03) -> sqrt((.0001+.0009)/4) = sqrt(.00025)
    assert downside_deviation(r, 0.0) == pytest.approx(math.sqrt(0.00025))
    expected = 0.0025 * 12 / (math.sqrt(0.00025) * math.sqrt(12))
    assert sortino_ratio(r, 0.0, 12) == pytest.approx(expected)


def test_max_drawdown_hand_example():
    r = pd.Series([0.10, -0.20, 0.05, -0.10, 0.50])
    # wealth 1.1, .88, .924, .8316, 1.2474 ; peak 1.1 ; worst .8316/1.1 - 1
    assert max_drawdown(r) == pytest.approx(0.8316 / 1.1 - 1)


def test_max_drawdown_counts_initial_capital():
    assert max_drawdown(pd.Series([-0.5, 0.1])) == pytest.approx(-0.5)


def test_cagr():
    r = pd.Series([0.01] * 24)
    assert cagr(r, 12) == pytest.approx(1.01**12 - 1)
