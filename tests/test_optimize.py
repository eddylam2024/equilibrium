"""Optimisation against closed-form solutions and constraint handling."""

from __future__ import annotations

import numpy as np
import pytest

from equilibrium.bl.equilibrium import equilibrium_returns
from equilibrium.bl.posterior import black_litterman
from equilibrium.core.errors import InfeasibleError, ValidationError
from equilibrium.core.types import ExpectedReturns, ReturnKind
from equilibrium.optimize.constraints import Constraints
from equilibrium.optimize.frontier import efficient_frontier
from equilibrium.optimize.optimizer import (
    Objective,
    max_sharpe,
    max_utility,
    min_variance,
    optimize,
    target_return,
    unconstrained_min_variance,
    unconstrained_tangency,
)

FREE = Constraints(long_only=False)


def _mu(cov3, values):
    return ExpectedReturns(cov3.tickers, np.array(values), kind=ReturnKind.EXCESS)


def test_min_variance_matches_closed_form(cov3):
    res = min_variance(cov3, FREE)
    np.testing.assert_allclose(res.weights.values, unconstrained_min_variance(cov3).values,
                               atol=1e-7)
    assert res.weights.total == pytest.approx(1.0)


def test_max_sharpe_matches_closed_form(cov3):
    mu = _mu(cov3, [0.06, 0.08, 0.03])
    res = max_sharpe(cov3, mu, FREE)
    np.testing.assert_allclose(res.weights.values, unconstrained_tangency(cov3, mu).values,
                               atol=1e-6)


def test_max_sharpe_is_optimal_under_constraints(cov3):
    mu = _mu(cov3, [0.06, 0.12, 0.01])
    cons = Constraints(long_only=True, max_weight=0.5)
    res = max_sharpe(cov3, mu, cons)
    assert res.weights.values.max() <= 0.5 + 1e-8 and res.weights.values.min() >= -1e-10
    rng = np.random.default_rng(1)
    for _ in range(2000):  # no random feasible portfolio beats it
        w = rng.dirichlet(np.ones(3))
        if w.max() > 0.5:
            continue
        s = w @ mu.values / np.sqrt(w @ cov3.values @ w)
        assert s <= res.sharpe + 1e-7


def test_bl_max_sharpe_without_views_recovers_market(cov3, w_mkt3):
    """Tangency of (Pi, (1+tau) Sigma) is proportional to Sigma^-1 Sigma w_mkt = w_mkt."""
    pi = equilibrium_returns(cov3, w_mkt3, 2.5)
    post = black_litterman(pi, cov3, np.zeros((0, 3)), np.zeros(0), np.zeros((0, 0)), 0.05)
    res = max_sharpe(post.predictive_cov, post.posterior, FREE)
    np.testing.assert_allclose(res.weights.values, w_mkt3.values, atol=1e-6)


def test_target_return_hits_target(cov3):
    mu = _mu(cov3, [0.06, 0.08, 0.03])
    res = target_return(cov3, mu, 0.065, Constraints())
    assert res.expected_excess_return == pytest.approx(0.065, abs=1e-8)
    # no lower-variance portfolio achieves the same return: compare with a nearby feasible one
    mv = min_variance(cov3, Constraints(), mu)
    assert res.volatility >= mv.volatility - 1e-10


def test_target_return_infeasible(cov3):
    mu = _mu(cov3, [0.06, 0.08, 0.03])
    with pytest.raises(InfeasibleError, match="achievable range"):
        target_return(cov3, mu, 0.20, Constraints(long_only=True))


def test_box_constraints_respected(cov3):
    res = min_variance(cov3, Constraints(long_only=True, min_weight=0.2, max_weight=0.4))
    assert np.all(res.weights.values >= 0.2 - 1e-9)
    assert np.all(res.weights.values <= 0.4 + 1e-9)


def test_invalid_constraints():
    with pytest.raises(ValueError):
        Constraints(min_weight=0.5, max_weight=0.2)
    with pytest.raises(ValueError):
        Constraints(long_only=True, min_weight=-0.1)
    with pytest.raises(InfeasibleError):
        Constraints(max_weight=0.2).bounds(3)  # 3 x 20% < 100%
    with pytest.raises(InfeasibleError):
        Constraints(min_weight=0.4).bounds(3)  # 3 x 40% > 100%


def test_max_sharpe_undefined_when_all_excess_returns_negative(cov3):
    with pytest.raises(InfeasibleError):
        max_sharpe(cov3, _mu(cov3, [-0.01, -0.02, -0.03]), Constraints())


def test_total_returns_rejected_by_optimizer(cov3):
    mu = ExpectedReturns(cov3.tickers, np.array([0.1, 0.1, 0.1]), kind=ReturnKind.TOTAL)
    with pytest.raises(ValidationError):
        max_sharpe(cov3, mu, Constraints())


def test_max_utility_matches_closed_form_when_budget_slack(cov3, w_mkt3):
    """With mu = delta Sigma w_mkt and a budget constraint, the optimum is w_mkt."""
    pi = equilibrium_returns(cov3, w_mkt3, 2.5)
    res = max_utility(cov3, pi, 2.5, FREE)
    np.testing.assert_allclose(res.weights.values, w_mkt3.values, atol=1e-6)


def test_dispatch_requires_inputs(cov3):
    with pytest.raises(ValidationError):
        optimize(Objective.MAX_SHARPE, cov3, None, Constraints())
    with pytest.raises(ValidationError):
        optimize(Objective.TARGET_RETURN, cov3, _mu(cov3, [0.05, 0.06, 0.07]), Constraints())


def test_frontier_is_monotone(cov3):
    mu = _mu(cov3, [0.06, 0.08, 0.03])
    f = efficient_frontier(cov3, mu, Constraints(), points=15)
    assert np.all(np.diff(f.returns) > -1e-9)
    assert np.all(np.diff(f.volatilities) > -1e-9)  # upper branch: vol rises with return
    assert f.max_sharpe is not None
    assert f.min_variance.volatility == pytest.approx(f.volatilities.min(), abs=1e-9)
