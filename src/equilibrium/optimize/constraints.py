"""Portfolio constraints and feasibility checks.

All portfolios are fully invested (sum w = 1). Box constraints ``lo <= w_i <= hi`` are
derived from ``long_only``, ``min_weight`` and ``max_weight``. Feasibility of the budget
constraint requires ``N * lo <= 1 <= N * hi``; feasibility of a target return is checked by
linear programming before any quadratic problem is solved.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.optimize import linprog

from equilibrium.core.errors import InfeasibleError
from equilibrium.core.types import FloatArray

UNBOUNDED_SHORT = -10.0  # default leverage cap when shorting is allowed, keeps QPs bounded


class Constraints(BaseModel):
    long_only: bool = True
    min_weight: float | None = Field(default=None, ge=-10.0, le=1.0)
    max_weight: float | None = Field(default=None, ge=-10.0, le=10.0)

    @model_validator(mode="after")
    def _check(self) -> Constraints:
        if self.long_only and self.min_weight is not None and self.min_weight < 0:
            raise ValueError("min_weight cannot be negative when long_only is set.")
        if (
            self.min_weight is not None
            and self.max_weight is not None
            and self.min_weight > self.max_weight
        ):
            raise ValueError(
                f"min_weight ({self.min_weight}) exceeds max_weight ({self.max_weight})."
            )
        return self

    def bounds(self, n: int) -> list[tuple[float, float]]:
        lo = self.min_weight if self.min_weight is not None else (
            0.0 if self.long_only else UNBOUNDED_SHORT)
        if self.long_only:
            lo = max(lo, 0.0)
        hi = self.max_weight if self.max_weight is not None else (
            1.0 if self.long_only else -UNBOUNDED_SHORT)
        if n * lo > 1.0 + 1e-12:
            raise InfeasibleError(
                f"min_weight {lo:.2%} x {n} assets = {n * lo:.2%} > 100%: the fully invested "
                "portfolio is infeasible."
            )
        if n * hi < 1.0 - 1e-12:
            raise InfeasibleError(
                f"max_weight {hi:.2%} x {n} assets = {n * hi:.2%} < 100%: the fully invested "
                "portfolio is infeasible."
            )
        return [(lo, hi)] * n

    def describe(self) -> str:
        parts = ["long-only" if self.long_only else "shorts allowed"]
        if self.min_weight is not None:
            parts.append(f"min {self.min_weight:.1%}")
        if self.max_weight is not None:
            parts.append(f"max {self.max_weight:.1%}")
        return ", ".join(parts)


def achievable_return_range(mu: FloatArray, bounds: list[tuple[float, float]]
                            ) -> tuple[float, float]:
    """Minimum and maximum of ``w' mu`` over the feasible set, by linear programming."""
    n = mu.size
    a_eq, b_eq = np.ones((1, n)), np.array([1.0])
    lo = linprog(mu, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi = linprog(-mu, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not (lo.success and hi.success):
        raise InfeasibleError("The constraint set is empty.")
    return float(lo.fun), float(-hi.fun)
