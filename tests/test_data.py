"""Data cleaning, providers and caching (offline: a fake provider stands in for Yahoo)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from equilibrium.core.errors import (
    DataError,
    InsufficientHistoryError,
    InvalidTickerError,
    ValidationError,
)
from equilibrium.data.cache import DataCache
from equilibrium.data.cleaning import clean_prices
from equilibrium.data.provider import PriceRequest
from equilibrium.data.synthetic import SyntheticProvider
from equilibrium.data.yahoo import YahooFinanceProvider


def _frame(data, start="2024-01-01"):
    return pd.DataFrame(data, index=pd.bdate_range(start, periods=len(next(iter(data.values())))))


def test_invalid_ticker_all_nan_raises():
    raw = _frame({"AAA": [1.0, 2.0, 3.0], "BAD": [np.nan] * 3})
    with pytest.raises(InvalidTickerError) as exc:
        clean_prices(raw, ("AAA", "BAD"))
    assert exc.value.tickers == ["BAD"]
    assert "never substituted" in str(exc.value)


def test_missing_column_raises():
    with pytest.raises(InvalidTickerError):
        clean_prices(_frame({"AAA": [1.0, 2.0, 3.0]}), ("AAA", "ZZZ"))


def test_duplicates_timezones_and_ordering():
    idx = pd.DatetimeIndex(["2024-01-03 05:00", "2024-01-02 05:00", "2024-01-03 05:00",
                            "2024-01-04 05:00"], tz="America/New_York")
    raw = pd.DataFrame({"A": [11.0, 10.0, 12.0, 13.0]}, index=idx)
    df, notes = clean_prices(raw, ("A",))
    assert df.index.tz is None
    assert list(df.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert df["A"].tolist() == [10.0, 12.0, 13.0]  # duplicate kept last
    assert any("duplicate" in n for n in notes)


def test_late_listing_trims_to_common_window():
    raw = _frame({"OLD": [1.0, 1.1, 1.2, 1.3, 1.4], "NEW": [np.nan, np.nan, 5.0, 5.1, 5.2]})
    df, notes = clean_prices(raw, ("OLD", "NEW"))
    assert len(df) == 3 and not df.isna().any().any()
    assert any("NEW" in n for n in notes)


def test_short_holiday_gap_forward_filled_and_disclosed():
    raw = _frame({"A": [1.0, 1.1, 1.2, 1.3], "B": [2.0, np.nan, 2.2, 2.3]})
    df, notes = clean_prices(raw, ("A", "B"))
    assert df.loc[df.index[1], "B"] == 2.0
    assert any("Forward-filled 1" in n for n in notes)


def test_long_gap_refused():
    raw = _frame({"A": list(np.linspace(1, 2, 12)), "B": [2.0] + [np.nan] * 8 + [2.1, 2.2, 2.3]})
    with pytest.raises(DataError, match="fabricate"):
        clean_prices(raw, ("A", "B"))


def test_insufficient_history():
    with pytest.raises(InsufficientHistoryError):
        clean_prices(_frame({"A": [1.0, 2.0]}), ("A",), min_observations=5)


def test_request_validation():
    with pytest.raises(ValidationError):
        PriceRequest(("A",), period="7y")
    with pytest.raises(ValidationError):
        PriceRequest(("A",), interval="1h")
    with pytest.raises(ValidationError):
        PriceRequest(("A",), start=dt.date(2024, 1, 2), end=dt.date(2024, 1, 1))
    r = PriceRequest(("A",), start=dt.date(2020, 1, 1))
    assert r.period is None  # explicit start overrides the default period


def test_synthetic_provider_is_deterministic_and_labelled():
    p = SyntheticProvider()
    req = PriceRequest(("AAA", "BBB"), period="2y")
    a, b = p.get_prices(req), p.get_prices(req)
    pd.testing.assert_frame_equal(a.prices, b.prices)
    assert a.synthetic and "SYNTHETIC" in a.notes[0]
    # a ticker's path does not depend on which other tickers were requested
    solo = p.get_prices(PriceRequest(("AAA",), period="2y"))
    pd.testing.assert_series_equal(solo.prices["AAA"], a.prices["AAA"])


class _FakeYahoo(YahooFinanceProvider):
    """Yahoo adapter with the network call replaced, to test caching and cleaning."""

    def __init__(self, cache, frame):
        super().__init__(cache)
        self.frame = frame
        self.calls = 0

    def _download(self, request):
        self.calls += 1
        return self.frame[list(request.tickers)]


def test_yahoo_adapter_caches(tmp_path):
    frame = _frame({"AAA": list(np.linspace(10, 20, 30)), "BBB": list(np.linspace(5, 6, 30))})
    prov = _FakeYahoo(DataCache(tmp_path), frame)
    req = PriceRequest(("AAA", "BBB"), period="1y")
    first = prov.get_prices(req)
    second = prov.get_prices(req)
    assert prov.calls == 1
    assert "cache" in second.notes[0]
    pd.testing.assert_frame_equal(first.prices, second.prices, check_freq=False)


def test_cache_disabled(tmp_path):
    frame = _frame({"AAA": list(np.linspace(10, 20, 30))})
    prov = _FakeYahoo(DataCache(tmp_path, enabled=False), frame)
    req = PriceRequest(("AAA",), period="1y")
    prov.get_prices(req)
    prov.get_prices(req)
    assert prov.calls == 2
