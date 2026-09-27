"""Mean-variance optimisation (Markowitz, 1952).

Objectives (all fully invested, subject to box constraints):

  min-variance     min  w' S w
  target-return    min  w' S w         s.t.  w' mu = target
  max-sharpe       max  (w' mu_x) / sqrt(w' S w)      with mu_x EXCESS returns
  max-utility      max  w' mu_x - (delta/2) w' S w

Max-Sharpe is not convex in w. We solve the equivalent convex quadratic programme of
Cornuejols & Tutuncu (2007), "Optimization Methods in Finance" (Cambridge Univ. Press):

  min  y' S y   s.t.  mu_x' y = 1,  1'y = kappa,  kappa >= 0,  lo*kappa <= y_i <= hi*kappa

and recover ``w = y / kappa``. The expected-return input must be *excess* returns; passing
total returns would subtract the risk-free rate twice (or not at all).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.optimize import minimize

from equilibrium.core.errors import InfeasibleError, NumericalError, ValidationError
from equilibrium.core.types import (
    CovarianceMatrix,
    ExpectedReturns,
    FloatArray,
    ReturnKind,
    Weights,
)
from equilibrium.optimize.constraints import Constraints, achievable_return_range
from equilibrium.risk.metrics import portfolio_volatility

SOLVER_TOL = 1e-12
FEAS_TOL = 1e-6


class Objective(StrEnum):
    MIN_VARIANCE = "min-variance"
    MAX_SHARPE = "max-sharpe"
    TARGET_RETURN = "target-return"
    MAX_UTILITY = "max-utility"


@dataclass(frozen=True)
class OptimizationResult:
    weights: Weights
    objective: Objective
    expected_excess_return: float | None
    volatility: float
    constraints: Constraints

    @property
    def sharpe(self) -> float | None:
        if self.expected_excess_return is None or self.volatility == 0:
            return None
        return self.expected_excess_return / self.volatility


def _solve_qp(n: int, fun, jac, x0: FloatArray, bounds, constraints) -> FloatArray:
    res = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints, method="SLSQP",
                   options={"ftol": SOLVER_TOL, "maxiter": 1000})
    if not res.success:
        raise NumericalError(f"Optimiser failed to converge: {res.message}")
    return np.asarray(res.x, dtype=np.float64)


def _check_feasible(w: FloatArray, bounds: list[tuple[float, float]]) -> FloatArray:
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    if abs(w.sum() - 1.0) > FEAS_TOL or np.any(w < lo - FEAS_TOL) or np.any(w > hi + FEAS_TOL):
        raise NumericalError("Optimiser returned a portfolio that violates the constraints.")
    w = np.clip(w, lo, hi)
    return w / w.sum()


def _excess(mu: ExpectedReturns, tickers: tuple[str, ...]) -> FloatArray:
    if mu.kind is not ReturnKind.EXCESS:
        raise ValidationError("Optimisation requires EXCESS expected returns.")
    return mu.aligned_to(tickers)


def _result(w: FloatArray, cov: CovarianceMatrix, mu: FloatArray | None, obj: Objective,
            cons: Constraints) -> OptimizationResult:
    return OptimizationResult(
        weights=Weights(cov.tickers, w),
        objective=obj,
        expected_excess_return=None if mu is None else float(w @ mu),
        volatility=portfolio_volatility(w, cov.values),
        constraints=cons,
    )


def _budget() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones_like(w)}


def min_variance(cov: CovarianceMatrix, constraints: Constraints,
                 mu: ExpectedReturns | None = None) -> OptimizationResult:
    s = cov.values
    n = s.shape[0]
    bounds = constraints.bounds(n)
    w = _solve_qp(n, lambda w: w @ s @ w, lambda w: 2 * s @ w, np.full(n, 1 / n), bounds,
                  [_budget()])
    w = _check_feasible(w, bounds)
    return _result(w, cov, None if mu is None else _excess(mu, cov.tickers),
                   Objective.MIN_VARIANCE, constraints)


def target_return(cov: CovarianceMatrix, mu: ExpectedReturns, target_excess: float,
                  constraints: Constraints) -> OptimizationResult:
    """Minimum-variance portfolio with ``w' mu = target`` (target in the units of ``mu``)."""
    m = _excess(mu, cov.tickers)
    s = cov.values
    n = s.shape[0]
    bounds = constraints.bounds(n)
    lo, hi = achievable_return_range(m, bounds)
    if not lo - 1e-10 <= target_excess <= hi + 1e-10:
        raise InfeasibleError(
            f"Target excess return {target_excess:.2%} is outside the achievable range "
            f"[{lo:.2%}, {hi:.2%}] under constraints ({constraints.describe()})."
        )
    cons = [_budget(), {"type": "eq", "fun": lambda w: w @ m - target_excess,
                        "jac": lambda w: m}]
    w = _solve_qp(n, lambda w: w @ s @ w, lambda w: 2 * s @ w, np.full(n, 1 / n), bounds, cons)
    w = _check_feasible(w, bounds)
    return _result(w, cov, m, Objective.TARGET_RETURN, constraints)


def max_sharpe(cov: CovarianceMatrix, mu: ExpectedReturns, constraints: Constraints
               ) -> OptimizationResult:
    """Tangency portfolio via the Cornuejols-Tutuncu convex reformulation."""
    m = _excess(mu, cov.tickers)
    s = cov.values
    n = s.shape[0]
    bounds = constraints.bounds(n)
    _, best = achievable_return_range(m, bounds)
    if best <= 1e-12:
        raise InfeasibleError(
            "No feasible portfolio has a positive expected excess return, so the maximum "
            "Sharpe ratio portfolio is undefined. Use min-variance instead."
        )
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    # variables z = (y_1..y_n, kappa)
    cons = [
        {"type": "eq", "fun": lambda z: z[:n] @ m - 1.0,
         "jac": lambda z: np.append(m, 0.0)},
        {"type": "eq", "fun": lambda z: np.sum(z[:n]) - z[n],
         "jac": lambda z: np.append(np.ones(n), -1.0)},
        {"type": "ineq", "fun": lambda z: z[:n] - lo * z[n],
         "jac": lambda z: np.hstack([np.eye(n), -lo[:, None]])},
        {"type": "ineq", "fun": lambda z: hi * z[n] - z[:n],
         "jac": lambda z: np.hstack([-np.eye(n), hi[:, None]])},
    ]
    # feasible start: scale the max-return LP vertex so that mu'y = 1
    w0 = np.full(n, 1 / n)
    if w0 @ m <= 0:
        w0 = min_variance(cov, constraints).weights.values
        if w0 @ m <= 0:
            w0 = _max_return_vertex(m, bounds)
    kappa0 = 1.0 / float(w0 @ m)
    z0 = np.append(w0 * kappa0, kappa0)
    obj = lambda z: z[:n] @ s @ z[:n]  # noqa: E731
    jac = lambda z: np.append(2 * s @ z[:n], 0.0)  # noqa: E731
    z = _solve_qp(n + 1, obj, jac, z0, [(None, None)] * n + [(0.0, None)], cons)
    if z[n] <= 1e-14:
        raise NumericalError("Max-Sharpe reformulation degenerated (kappa = 0).")
    w = _check_feasible(z[:n] / z[n], bounds)
    return _result(w, cov, m, Objective.MAX_SHARPE, constraints)


def _max_return_vertex(m: FloatArray, bounds: list[tuple[float, float]]) -> FloatArray:
    from scipy.optimize import linprog

    res = linprog(-m, A_eq=np.ones((1, m.size)), b_eq=[1.0], bounds=bounds, method="highs")
    return np.asarray(res.x)


def max_utility(cov: CovarianceMatrix, mu: ExpectedReturns, risk_aversion: float,
                constraints: Constraints) -> OptimizationResult:
    m = _excess(mu, cov.tickers)
    s = cov.values
    n = s.shape[0]
    if risk_aversion <= 0:
        raise ValidationError("Risk aversion must be positive.")
    bounds = constraints.bounds(n)
    d = risk_aversion
    w = _solve_qp(n, lambda w: -(w @ m) + 0.5 * d * (w @ s @ w), lambda w: -m + d * s @ w,
                  np.full(n, 1 / n), bounds, [_budget()])
    w = _check_feasible(w, bounds)
    return _result(w, cov, m, Objective.MAX_UTILITY, constraints)


def optimize(objective: Objective, cov: CovarianceMatrix, mu: ExpectedReturns | None,
             constraints: Constraints, *, target: float | None = None,
             risk_aversion: float | None = None) -> OptimizationResult:
    """Dispatch to an objective, validating that its required inputs were provided."""
    if objective is Objective.MIN_VARIANCE:
        return min_variance(cov, constraints, mu)
    if mu is None:
        raise ValidationError(f"Objective {objective.value} requires expected returns.")
    if objective is Objective.MAX_SHARPE:
        return max_sharpe(cov, mu, constraints)
    if objective is Objective.TARGET_RETURN:
        if target is None:
            raise ValidationError("target-return objective requires --target-return.")
        return target_return(cov, mu, target, constraints)
    if risk_aversion is None:
        raise ValidationError("max-utility objective requires a risk aversion.")
    return max_utility(cov, mu, risk_aversion, constraints)


def unconstrained_tangency(cov: CovarianceMatrix, mu: ExpectedReturns) -> Weights:
    """Closed form ``w ∝ S^-1 mu`` normalised to sum to one (used for validation)."""
    m = _excess(mu, cov.tickers)
    raw = np.linalg.solve(cov.values, m)
    if raw.sum() <= 0:
        raise InfeasibleError("Tangency portfolio undefined: 1' S^-1 mu <= 0.")
    return Weights(cov.tickers, raw / raw.sum())


def unconstrained_min_variance(cov: CovarianceMatrix) -> Weights:
    """Closed form ``w = S^-1 1 / (1' S^-1 1)`` (used for validation)."""
    raw = np.linalg.solve(cov.values, np.ones(len(cov.tickers)))
    return Weights(cov.tickers, raw / raw.sum())
