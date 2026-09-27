"""Core value types shared by every layer.

The central design rule: every vector of per-asset numbers carries the ticker ordering it
is aligned to, and every vector of *returns* additionally declares whether it is an excess
or a total return and whether it is an historical observation or a model estimate. Mixing
excess and total returns is the single most common Black-Litterman implementation bug,
so the types make it impossible to do silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

from equilibrium.core.errors import ValidationError

FloatArray = npt.NDArray[np.float64]

WEIGHT_SUM_TOL = 1e-8


class Frequency(StrEnum):
    """Sampling frequency of a return series."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def periods_per_year(self) -> int:
        """Conventional annualisation factor (252 trading days, 52 weeks, 12 months)."""
        return {"daily": 252, "weekly": 52, "monthly": 12}[self.value]

    @property
    def pandas_rule(self) -> str | None:
        """Resampling rule for pandas; ``None`` means keep the native trading-day index."""
        return {"daily": None, "weekly": "W-FRI", "monthly": "ME"}[self.value]


class ReturnKind(StrEnum):
    """Whether a return is measured in excess of the risk-free rate."""

    EXCESS = "excess"
    TOTAL = "total"


class Source(StrEnum):
    """Epistemic status of a quantity: observed in history, or produced by a model."""

    HISTORICAL = "historical"
    MODEL = "model"


def normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    """Upper-case, strip and validate ticker symbols; reject duplicates and unsafe input.

    Tickers are only ever used as data keys and are never executed or interpolated into
    shell commands, but we still restrict them to the character set used by real symbols.
    """
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^=")
    out: list[str] = []
    for raw in tickers:
        t = raw.strip().upper()
        if not t or len(t) > 15 or any(c not in allowed for c in t):
            raise ValidationError(f"Invalid ticker symbol: {raw!r}")
        if t in out:
            raise ValidationError(f"Duplicate ticker: {t}")
        out.append(t)
    if not out:
        raise ValidationError("At least one ticker is required.")
    return tuple(out)


@dataclass(frozen=True)
class AssetVector:
    """A real-valued vector aligned to an explicit ticker ordering."""

    tickers: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.tickers):
            raise ValidationError(
                f"Vector has {values.shape[0]} entries but {len(self.tickers)} tickers."
            )
        if not np.all(np.isfinite(values)):
            raise ValidationError("Vector contains NaN or infinite values.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.tickers)

    def __getitem__(self, ticker: str) -> float:
        return float(self.values[self.tickers.index(ticker)])

    def to_series(self, name: str | None = None) -> pd.Series:
        return pd.Series(self.values, index=list(self.tickers), name=name)

    def aligned_to(self, tickers: Sequence[str]) -> FloatArray:
        """Return values reordered to ``tickers``; raises if the ticker sets differ."""
        if set(tickers) != set(self.tickers) or len(tickers) != len(self.tickers):
            raise ValidationError(
                f"Ticker mismatch: {list(self.tickers)} vs {list(tickers)}"
            )
        idx = [self.tickers.index(t) for t in tickers]
        return self.values[idx]


@dataclass(frozen=True)
class Weights(AssetVector):
    """Portfolio weights. Fully-invested portfolios sum to one (checked on request)."""

    def __post_init__(self) -> None:
        super().__post_init__()

    @classmethod
    def normalized(cls, tickers: Sequence[str], raw: npt.ArrayLike) -> Weights:
        """Scale non-negative raw values (e.g. market capitalisations) to sum to one."""
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if np.any(arr < 0):
            raise ValidationError("Cannot normalise negative values into weights.")
        total = arr.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValidationError("Cannot normalise weights: values sum to zero.")
        return cls(tuple(tickers), arr / total)

    @classmethod
    def equal(cls, tickers: Sequence[str]) -> Weights:
        n = len(tickers)
        return cls(tuple(tickers), np.full(n, 1.0 / n))

    @property
    def total(self) -> float:
        return float(self.values.sum())

    def require_fully_invested(self, tol: float = WEIGHT_SUM_TOL) -> None:
        if abs(self.total - 1.0) > tol:
            raise ValidationError(f"Weights sum to {self.total:.10f}, expected 1.")


@dataclass(frozen=True)
class ExpectedReturns(AssetVector):
    """Annualised expected returns with explicit kind (excess/total) and source."""

    kind: ReturnKind = ReturnKind.EXCESS
    source: Source = Source.MODEL
    label: str = ""

    def to_excess(self, risk_free_rate: float) -> ExpectedReturns:
        if self.kind is ReturnKind.EXCESS:
            return self
        return ExpectedReturns(
            self.tickers, self.values - risk_free_rate, ReturnKind.EXCESS, self.source, self.label
        )

    def to_total(self, risk_free_rate: float) -> ExpectedReturns:
        if self.kind is ReturnKind.TOTAL:
            return self
        return ExpectedReturns(
            self.tickers, self.values + risk_free_rate, ReturnKind.TOTAL, self.source, self.label
        )


@dataclass(frozen=True)
class CovarianceMatrix:
    """Annualised covariance matrix aligned to a ticker ordering."""

    tickers: tuple[str, ...]
    values: FloatArray
    source: Source = Source.HISTORICAL
    estimator: str = "sample"
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = np.asarray(self.values, dtype=np.float64)
        n = len(self.tickers)
        if m.shape != (n, n):
            raise ValidationError(f"Covariance shape {m.shape} does not match {n} tickers.")
        if not np.all(np.isfinite(m)):
            raise ValidationError("Covariance contains NaN or infinite values.")
        m = m.copy()
        m.setflags(write=False)
        object.__setattr__(self, "values", m)

    @property
    def volatilities(self) -> FloatArray:
        return np.sqrt(np.clip(np.diag(self.values), 0.0, None))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values, index=list(self.tickers), columns=list(self.tickers))
