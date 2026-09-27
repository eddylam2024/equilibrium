"""Live Yahoo Finance integration tests. Deselected by default; run with ``pytest -m live``."""

from __future__ import annotations

import pytest

from equilibrium.core.errors import DataError, InvalidTickerError
from equilibrium.data.cache import DataCache
from equilibrium.data.provider import PriceRequest
from equilibrium.data.yahoo import YahooFinanceProvider

pytestmark = pytest.mark.live


@pytest.fixture
def yahoo(tmp_path):
    return YahooFinanceProvider(DataCache(tmp_path))


def test_live_prices_and_caps(yahoo):
    data = yahoo.get_prices(PriceRequest(("AAPL", "MSFT"), period="1y"))
    assert len(data.prices) > 200 and not data.prices.isna().any().any()
    assert data.prices.index.tz is None and data.adjusted
    caps = yahoo.get_market_caps(["AAPL", "MSFT"])
    assert all(c > 1e11 for c in caps.values())


def test_live_invalid_ticker(yahoo):
    with pytest.raises(InvalidTickerError):
        yahoo.get_prices(PriceRequest(("AAPL", "ZZZZNOTREAL123"), period="1y"))


def test_live_etf_has_no_market_cap(yahoo):
    with pytest.raises(DataError, match="market-weights"):
        yahoo.get_market_caps(["SPY"])
