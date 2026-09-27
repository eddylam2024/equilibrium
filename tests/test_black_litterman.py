"""Black-Litterman validation.

The central test uses a 3-asset example worked by hand below; the expected numbers are
written as exact fractions derived on paper, not produced by the implementation.

  Sigma : vols (20%, 30%, 15%), pairwise corr 0.2
          [[.0400, .0120, .0060], [.0120, .0900, .0090], [.0060, .0090, .0225]]
  w_mkt : (0.5, 0.3, 0.2)       delta = 2.5       tau = 0.05
  Sigma w_mkt = (.0248, .0348, .0102)
  Pi = 2.5 * Sigma w_mkt = (.062, .087, .0255)

  View : asset C has excess return q = 5%, omega = 0.0005, p = (0, 0, 1)
  tau Sigma p'      = (.0003, .00045, .001125)
  p tau Sigma p'    = .001125
  q - p Pi          = .05 - .0255 = .0245
  scale             = .0245 / (.001125 + .0005) = .0245 / .001625 = 196/13
  mu_BL = Pi + tau Sigma p' * 196/13
        = (.062 + .0588/13, .087 + .0882/13, .0255 + .2205/13)
  check: mu_C is the precision-weighted average  (4/13) * .0255 + (9/13) * .05  ✓
  M_CC = .001125 - .001125^2 / .001625 = .001125 * 4/13
"""

from __future__ import annotations

import numpy as np
import pytest

from equilibrium.bl.equilibrium import (
    equilibrium_returns,
    implied_risk_aversion,
    market_cap_weights,
)
from equilibrium.bl.omega import build_omega, idzorek_omega, implied_confidence
from equilibrium.bl.posterior import (
    black_litterman,
    posterior_mean_canonical,
    posterior_mean_update,
)
from equilibrium.bl.views import View, ViewType, build_view_matrices
from equilibrium.core.errors import NumericalError, ValidationError
from equilibrium.core.types import CovarianceMatrix, ExpectedReturns, ReturnKind, Weights

TAU = 0.05
DELTA = 2.5


def _pi(cov3, w_mkt3):
    return equilibrium_returns(cov3, w_mkt3, DELTA)


# ------------------------------------------------------------------ equilibrium
def test_equilibrium_returns_hand_example(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3)
    np.testing.assert_allclose(pi.values, [0.062, 0.087, 0.0255], atol=1e-15)
    assert pi.kind is ReturnKind.EXCESS


def test_equilibrium_uses_covariance_not_correlation(cov3, w_mkt3):
    from equilibrium.risk.covariance import correlation_from_covariance

    corr = CovarianceMatrix(cov3.tickers, correlation_from_covariance(cov3.values))
    assert not np.allclose(_pi(cov3, w_mkt3).values, equilibrium_returns(corr, w_mkt3,
                                                                         DELTA).values)


def test_market_cap_weights_sum_to_one():
    w = market_cap_weights({"A": 3e12, "B": 1e12, "C": 1e12}, ["A", "B", "C"])
    assert w.total == pytest.approx(1.0)
    np.testing.assert_allclose(w.values, [0.6, 0.2, 0.2])


def test_market_cap_weights_errors():
    with pytest.raises(ValidationError):
        market_cap_weights({"A": 1.0}, ["A", "B"])
    with pytest.raises(ValidationError):
        market_cap_weights({"A": 1.0, "B": -1.0}, ["A", "B"])


def test_unnormalised_market_weights_rejected(cov3):
    with pytest.raises(ValidationError):
        equilibrium_returns(cov3, Weights(cov3.tickers, np.array([0.5, 0.5, 0.5])), DELTA)


def test_implied_risk_aversion():
    assert implied_risk_aversion(0.06, 0.16**2) == pytest.approx(0.06 / 0.0256)


# ------------------------------------------------------------------ views
def test_absolute_view_converted_from_total_to_excess():
    v = View(type=ViewType.ABSOLUTE, asset="ccc", value=0.08)
    P, Q = build_view_matrices([v], ["AAA", "BBB", "CCC"], risk_free_rate=0.03)
    np.testing.assert_array_equal(P, [[0, 0, 1]])
    np.testing.assert_allclose(Q, [0.05])


def test_relative_view_needs_no_risk_free_adjustment():
    v = View(type=ViewType.RELATIVE, long="CCC", short="BBB", value=0.05)
    P, Q = build_view_matrices([v], ["AAA", "BBB", "CCC"], risk_free_rate=0.03)
    np.testing.assert_array_equal(P, [[0, -1, 1]])
    np.testing.assert_allclose(Q, [0.05])


def test_multiple_views_matrix():
    views = [
        View(type=ViewType.ABSOLUTE, asset="AAA", value=0.10),
        View(type=ViewType.RELATIVE, long="CCC", short="BBB", value=0.02),
    ]
    P, Q = build_view_matrices(views, ["AAA", "BBB", "CCC"], 0.0)
    np.testing.assert_array_equal(P, [[1, 0, 0], [0, -1, 1]])
    np.testing.assert_allclose(Q, [0.10, 0.02])


def test_view_validation():
    with pytest.raises(ValueError):
        View(type=ViewType.RELATIVE, long="A", short="A", value=0.01)
    with pytest.raises(ValueError):
        View(type=ViewType.ABSOLUTE, asset="A", value=8.0)  # 800%: almost surely a % typo
    with pytest.raises(ValueError):
        View(type=ViewType.ABSOLUTE, asset="A", value=0.1, confidence=0.0)
    with pytest.raises(ValidationError):
        build_view_matrices([View(type=ViewType.ABSOLUTE, asset="ZZZ", value=0.1)],
                            ["AAA"], 0.0)


# ------------------------------------------------------------------ the hand-verified posterior
def test_posterior_hand_example(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3)
    P = np.array([[0.0, 0.0, 1.0]])
    Q = np.array([0.05])
    omega = np.array([[0.0005]])
    res = black_litterman(pi, cov3, P, Q, omega, TAU)
    expected = np.array([0.062 + 0.0588 / 13, 0.087 + 0.0882 / 13, 0.0255 + 0.2205 / 13])
    np.testing.assert_allclose(res.posterior.values, expected, rtol=0, atol=1e-14)
    assert res.posterior.values[2] == pytest.approx(4 / 13 * 0.0255 + 9 / 13 * 0.05, abs=1e-14)
    assert res.mean_uncertainty[2, 2] == pytest.approx(0.001125 * 4 / 13, abs=1e-15)
    np.testing.assert_allclose(res.predictive_cov.values, cov3.values + res.mean_uncertainty)


def test_canonical_and_update_forms_agree(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3).values
    P = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]])
    Q = np.array([0.04, 0.03])
    omega = np.diag([0.0004, 0.0009])
    a = posterior_mean_canonical(pi, cov3.values, P, Q, omega, TAU)
    b = posterior_mean_update(pi, cov3.values, P, Q, omega, TAU)
    np.testing.assert_allclose(a, b, atol=1e-13)


def test_posterior_uncertainty_forms_agree(cov3, w_mkt3):
    P = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 1.0]])
    omega = np.diag([0.0004, 0.0009])
    res = black_litterman(_pi(cov3, w_mkt3), cov3, P, np.array([0.04, 0.03]), omega, TAU)
    precision = np.linalg.inv(TAU * cov3.values) + P.T @ np.linalg.inv(omega) @ P
    np.testing.assert_allclose(res.mean_uncertainty, np.linalg.inv(precision), atol=1e-14)


# ------------------------------------------------------------------ limiting behaviour
def test_no_views_returns_equilibrium_exactly(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3)
    res = black_litterman(pi, cov3, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)), TAU)
    np.testing.assert_array_equal(res.posterior.values, pi.values)
    np.testing.assert_allclose(res.mean_uncertainty, TAU * cov3.values)
    np.testing.assert_allclose(res.predictive_cov.values, (1 + TAU) * cov3.values)


def test_no_views_unconstrained_weights_are_market_over_one_plus_tau(cov3, w_mkt3):
    """He & Litterman (1999): w* = (delta Sigma_post)^-1 Pi = w_mkt / (1 + tau)."""
    pi = _pi(cov3, w_mkt3)
    res = black_litterman(pi, cov3, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)), TAU)
    w = np.linalg.solve(DELTA * res.predictive_cov.values, res.posterior.values)
    np.testing.assert_allclose(w, w_mkt3.values / (1 + TAU), atol=1e-14)


def _single_view_shift(cov3, w_mkt3, omega_value):
    pi = _pi(cov3, w_mkt3)
    res = black_litterman(pi, cov3, np.array([[0.0, 0.0, 1.0]]), np.array([0.05]),
                          np.array([[omega_value]]), TAU)
    return res.posterior.values - pi.values


def test_near_zero_confidence_has_minimal_influence(cov3, w_mkt3):
    prior_var = 0.05 * 0.0225
    shift = _single_view_shift(cov3, w_mkt3, idzorek_omega(1e-9, prior_var))
    assert np.max(np.abs(shift)) < 1e-10


def test_confidence_is_monotone_and_full_confidence_hits_view(cov3, w_mkt3):
    prior_var = 0.05 * 0.0225
    shifts = [abs(_single_view_shift(cov3, w_mkt3, idzorek_omega(c, prior_var))[2])
              for c in (0.1, 0.5, 0.9, 1.0)]
    assert shifts == sorted(shifts)
    pi = _pi(cov3, w_mkt3)
    res = black_litterman(pi, cov3, np.array([[0.0, 0.0, 1.0]]), np.array([0.05]),
                          np.zeros((1, 1)), TAU)  # 100% confidence: Omega = 0
    assert res.posterior.values[2] == pytest.approx(0.05, abs=1e-14)
    assert res.diagnostics["canonical_vs_update_max_abs_diff"].startswith("n/a")


@pytest.mark.parametrize("confidence", [0.1, 0.3, 0.7, 0.95])
def test_idzorek_confidence_is_fraction_of_full_tilt(cov3, w_mkt3, confidence):
    """Idzorek (2005) definition, checked numerically on unconstrained weights."""
    pi = _pi(cov3, w_mkt3)
    P, Q = np.array([[0.0, -1.0, 1.0]]), np.array([0.03])

    def weights(omega):
        res = black_litterman(pi, cov3, P, Q, np.array([[omega]]), TAU)
        return np.linalg.solve(DELTA * cov3.values, res.posterior.values)

    v = View(type=ViewType.RELATIVE, long="CCC", short="BBB", value=0.03, confidence=confidence)
    omega, methods = build_omega([v], P, cov3.values, TAU)
    tilt = weights(omega[0, 0]) - w_mkt3.values
    full = weights(0.0) - w_mkt3.values
    np.testing.assert_allclose(tilt, confidence * full, atol=1e-12)
    assert methods[0].value == "idzorek"
    assert implied_confidence(omega[0, 0], float((P @ (TAU * cov3.values) @ P.T)[0, 0])) == \
        pytest.approx(confidence)


def test_he_litterman_default_omega_and_tau_invariance(cov3, w_mkt3):
    v = View(type=ViewType.ABSOLUTE, asset="CCC", value=0.05)
    P, Q = build_view_matrices([v], cov3.tickers, 0.0)
    posts = []
    for tau in (0.025, 0.05, 0.5):
        omega, _ = build_omega([v], P, cov3.values, tau)
        assert omega[0, 0] == pytest.approx(tau * 0.0225)
        posts.append(black_litterman(_pi(cov3, w_mkt3), cov3, P, Q, omega, tau)
                     .posterior.values)
    np.testing.assert_allclose(posts[0], posts[1], atol=1e-14)
    np.testing.assert_allclose(posts[0], posts[2], atol=1e-14)


def test_view_equal_to_prior_changes_nothing(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3)
    res = black_litterman(pi, cov3, np.array([[0.0, 0.0, 1.0]]), np.array([pi.values[2]]),
                          np.array([[0.0001]]), TAU)
    np.testing.assert_allclose(res.posterior.values, pi.values, atol=1e-15)


def test_contradictory_certain_views_raise(cov3, w_mkt3):
    P = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    with pytest.raises(NumericalError):
        black_litterman(_pi(cov3, w_mkt3), cov3, P, np.array([0.05, 0.07]),
                        np.zeros((2, 2)), TAU)


def test_total_return_prior_rejected(cov3, w_mkt3):
    pi = _pi(cov3, w_mkt3)
    total = ExpectedReturns(pi.tickers, pi.values, kind=ReturnKind.TOTAL)
    with pytest.raises(ValidationError):
        black_litterman(total, cov3, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)), TAU)
