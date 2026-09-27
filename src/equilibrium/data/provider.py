"""Abstract market-data interface. The quantitative engine depends only on this contract."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from equilibrium.core.errors import ValidationError
from equilibrium.core.types import Frequency
from equilibrium.returns.calc import ReturnMethod, periodic_returns

VALID_PERIODS = ("1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "max")
VALID_INTERVALS = ("1d", "1wk", "1mo")


@dataclass(frozen=True)
class PriceRequest:
    """Which prices to fetch. Either ``period`` or ``start`` (optionally with ``end``)."""

    tickers: tuple[str, ...]
    start: dt.date | None = None
    end: dt.date | None = None
    period: str | None = "10y"
    interval: str = "1d"

    def __post_init__(self) -> None:
        if self.start is not None:
            object.__setattr__(self, "period", None)
        if self.period is None and self.start is None:
            raise ValidationError("Provide either a period or a start date.")
        if self.period is not None and self.period not in VALID_PERIODS:
            raise ValidationError(f"Invalid period {self.period!r}; use one of {VALID_PERIODS}.")
        if self.interval not in VALID_INTERVALS:
            raise ValidationError(
                f"Invalid interval {self.interval!r}; use one of {VALID_INTERVALS}."
            )
        if self.start and self.end and self.start >= self.end:
            raise ValidationError("start must be before end.")
        if self.end and self.end > dt.date.today() + dt.timedelta(days=1):
            raise ValidationError("end date is in the future.")


@dataclass(frozen=True)
class AssetMetadata:
    ticker: str
    name: str | None = None
    currency: str | None = None
    exchange: str | None = None
    quote_type: str | None = None


@dataclass
class PriceData:
    """Cleaned, aligned adjusted prices plus a record of every cleaning decision."""

    prices: pd.DataFrame
    source: str
    adjusted: bool
    notes: list[str] = field(default_factory=list)
    synthetic: bool = False

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.prices.columns)


class MarketDataProvider(ABC):
    """Replaceable source of market data (Yahoo Finance, a vendor feed, synthetic data...)."""

    name: str = "abstract"
    synthetic: bool = False

    @abstractmethod
    def get_prices(self, request: PriceRequest) -> PriceData:
        """Adjusted close prices (total-return basis where the source supports it)."""

    @abstractmethod
    def get_market_caps(self, tickers: Sequence[str]) -> dict[str, float]:
        """Latest market capitalisation per ticker, in each security's quote currency."""

    @abstractmethod
    def get_metadata(self, tickers: Sequence[str]) -> dict[str, AssetMetadata]:
        """Descriptive metadata (name, currency, exchange)."""

    def get_returns(self, request: PriceRequest, frequency: Frequency,
                    method: ReturnMethod = ReturnMethod.SIMPLE) -> pd.DataFrame:
        return periodic_returns(self.get_prices(request).prices, frequency, method)
