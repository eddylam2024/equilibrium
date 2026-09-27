"""Investor views and their matrix representation ``P mu = Q + eps,  eps ~ N(0, Omega)``.

Absolute view   "NVDA will return 8%"           P row = e_NVDA,            Q = 0.08 - r_f
Relative view   "NVDA will beat MSFT by 5%"     P row = e_NVDA - e_MSFT,   Q = 0.05

Absolute views are entered by users as *total* annual returns (that is how people think)
and converted to excess returns here, because ``Pi`` and the posterior are excess
returns. Relative views need no conversion because the risk-free rate cancels.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field, model_validator

from equilibrium.core.errors import ValidationError
from equilibrium.core.types import FloatArray


class ViewType(StrEnum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class View(BaseModel):
    """A single investor view. Returns are annual decimals (0.05 = 5%)."""

    type: ViewType
    asset: str | None = None
    long: str | None = None
    short: str | None = None
    value: float = Field(description="Absolute: total annual return. Relative: annual spread.")
    confidence: float | None = Field(
        default=None, gt=0.0, le=1.0, description="Idzorek confidence in (0, 1]."
    )
    omega: float | None = Field(default=None, ge=0.0, description="Explicit view variance.")

    @model_validator(mode="after")
    def _check(self) -> View:
        if self.type is ViewType.ABSOLUTE:
            if not self.asset or self.long or self.short:
                raise ValueError("An absolute view needs exactly one asset.")
            self.asset = self.asset.upper()
        else:
            if not self.long or not self.short or self.asset:
                raise ValueError("A relative view needs a long and a short asset.")
            self.long, self.short = self.long.upper(), self.short.upper()
            if self.long == self.short:
                raise ValueError("Long and short assets must differ.")
        if not -1.0 < self.value < 5.0:
            raise ValueError(
                f"View return {self.value} is outside (-100%, 500%); returns are decimals "
                "(use 0.05 for 5%)."
            )
        if self.confidence is not None and self.omega is not None:
            raise ValueError("Specify either a confidence or an explicit omega, not both.")
        return self

    @property
    def assets(self) -> tuple[str, ...]:
        if self.type is ViewType.ABSOLUTE:
            return (str(self.asset),)
        return (str(self.long), str(self.short))

    def describe(self) -> str:
        if self.type is ViewType.ABSOLUTE:
            return f"{self.asset} returns {self.value:+.2%}"
        return f"{self.long} ≻ {self.short} by {self.value:+.2%}"


def build_view_matrices(
    views: Sequence[View], tickers: Sequence[str], risk_free_rate: float
) -> tuple[FloatArray, FloatArray]:
    """Return the pick matrix ``P`` (K x N) and excess view returns ``Q`` (K,)."""
    if not views:
        return np.zeros((0, len(tickers))), np.zeros(0)
    index = {t: i for i, t in enumerate(tickers)}
    P = np.zeros((len(views), len(tickers)))
    Q = np.zeros(len(views))
    for k, v in enumerate(views):
        unknown = [a for a in v.assets if a not in index]
        if unknown:
            raise ValidationError(
                f"View '{v.describe()}' references {', '.join(unknown)}, which is not in the "
                f"universe ({', '.join(tickers)})."
            )
        if v.type is ViewType.ABSOLUTE:
            P[k, index[str(v.asset)]] = 1.0
            Q[k] = v.value - risk_free_rate
        else:
            P[k, index[str(v.long)]] = 1.0
            P[k, index[str(v.short)]] = -1.0
            Q[k] = v.value
    return P, Q
