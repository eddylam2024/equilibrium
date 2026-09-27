"""Efficient frontier: minimum-variance portfolios across a grid of target returns.

The frontier is only as meaningful as its expected-return input. When ``mu`` is a historical
sample mean the curve describes the past, not a forecast; callers must label it accordingly
(``ExpectedReturns.source``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from equilibrium.core.errors import EquilibriumError
from equilibrium.core.types import CovarianceMatrix, ExpectedReturns, Source
from equilibrium.optimize.constraints import Constraints
from equilibrium.optimize.optimizer import (
    OptimizationResult,
    max_sharpe,
    min_variance,
    target_return,
)


@dataclass(frozen=True)
class Frontier:
    returns: np.ndarray
    volatilities: np.ndarray
    portfolios: list[OptimizationResult]
    min_variance: OptimizationResult
    max_sharpe: OptimizationResult | None
    source: Source


def efficient_frontier(cov: CovarianceMatrix, mu: ExpectedReturns, constraints: Constraints,
                       points: int = 40) -> Frontier:
    mv = min_variance(cov, constraints, mu)
    m = mu.aligned_to(cov.tickers)
    bounds = constraints.bounds(len(cov.tickers))
    from equilibrium.optimize.constraints import achievable_return_range

    _, r_max = achievable_return_range(m, bounds)
    r_min = float(mv.expected_excess_return or 0.0)
    targets = np.linspace(r_min, r_max, points)
    portfolios: list[OptimizationResult] = [mv]
    for t in targets[1:]:
        try:
            portfolios.append(target_return(cov, mu, float(t), constraints))
        except EquilibriumError:
            continue  # grid point at the numerical edge of feasibility
    try:
        ms: OptimizationResult | None = max_sharpe(cov, mu, constraints)
    except EquilibriumError:
        ms = None
    rets = np.array([p.expected_excess_return for p in portfolios], dtype=float)
    vols = np.array([p.volatility for p in portfolios])
    return Frontier(rets, vols, portfolios, mv, ms, mu.source)
