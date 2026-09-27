"""Yahoo Finance adapter. The only module that imports ``yfinance``.

Prices are requested with ``auto_adjust=True``: closes are adjusted for splits and cash
dividends, so percentage changes approximate total returns. Yahoo's data is unofficial and
unaudited; it is suitable for research and education, not for production trading.

Market capitalisation is Yahoo's *current* value (``fast_info.market_cap``). Yahoo does not
provide historical market capitalisations; see ``backtest.strategies`` for how that
limitation is handled and disclosed.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import logging
from collections.abc import Sequence

import pandas as pd

from equilibrium.core.errors import DataError, InvalidTickerError
from equilibrium.data.cache import DataCache
from equilibrium.data.cleaning import clean_prices
from equilibrium.data.provider import AssetMetadata, MarketDataProvider, PriceData, PriceRequest

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_yfinance():
    """yfinance prints failures to stdout/stderr; we surface them as exceptions instead."""
    yf_log = logging.getLogger("yfinance")
    previous = yf_log.level
    yf_log.setLevel(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        yf_log.setLevel(previous)


class YahooFinanceProvider(MarketDataProvider):
    name = "Yahoo Finance (yfinance)"

    def __init__(self, cache: DataCache | None = None) -> None:
        self.cache = cache or DataCache()

    def _download(self, request: PriceRequest) -> pd.DataFrame:
        import yfinance as yf

        kwargs: dict[str, object] = {"interval": request.interval}
        if request.period:
            kwargs["period"] = request.period
        else:
            kwargs["start"] = request.start.isoformat() if request.start else None
            end = request.end
            if end is not None:
                end = end + dt.timedelta(days=1)  # yfinance's end is exclusive
            kwargs["end"] = end.isoformat() if end else None
        try:
            with _quiet_yfinance():
                raw = yf.download(list(request.tickers), auto_adjust=True, progress=False,
                                  threads=True, group_by="column", **kwargs)
        except Exception as exc:  # network errors, rate limits, parsing failures
            raise DataError(f"Yahoo Finance download failed: {exc}") from exc
        if raw is None or raw.empty:
            raise InvalidTickerError(list(request.tickers),
                                     "Yahoo Finance returned no data (check network access).")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:  # single ticker, flat columns
            close = raw[["Close"]].rename(columns={"Close": request.tickers[0]})
        return close

    def get_prices(self, request: PriceRequest) -> PriceData:
        historical_only = request.end is not None and request.end < dt.date.today()
        key = self.cache.key("prices", request.tickers, request.start, request.end,
                             request.period, request.interval, dt.date.today()
                             if not historical_only else None)
        cached = self.cache.get_frame(key, permanent=historical_only)
        notes: list[str] = []
        if cached is not None:
            raw = cached
            notes.append("Loaded from local cache.")
        else:
            raw = self._download(request)
            self.cache.put_frame(key, raw)
        prices, clean_notes = clean_prices(raw, request.tickers)
        return PriceData(prices, source=self.name, adjusted=True, notes=notes + clean_notes)

    def get_market_caps(self, tickers: Sequence[str]) -> dict[str, float]:
        import yfinance as yf

        key = self.cache.key("mcap", tuple(tickers), dt.date.today())
        cached = self.cache.get_json(key)
        if cached is not None:
            return {k: float(v) for k, v in cached.items()}
        caps: dict[str, float] = {}
        missing: list[str] = []
        for t in tickers:
            try:
                with _quiet_yfinance():
                    cap = yf.Ticker(t).fast_info.market_cap
            except Exception:  # yfinance raises assorted errors for unknown symbols
                cap = None
            if cap is None or not cap > 0:
                missing.append(t)
            else:
                caps[t] = float(cap)
        if missing:
            raise DataError(
                "No market capitalisation available for " + ", ".join(missing)
                + " (ETFs, indices and funds have none). Provide weights explicitly with "
                "--market-weights, e.g. --market-weights SPY=0.6,TLT=0.4."
            )
        self.cache.put_json(key, caps)
        return caps

    def get_metadata(self, tickers: Sequence[str]) -> dict[str, AssetMetadata]:
        import yfinance as yf

        out: dict[str, AssetMetadata] = {}
        for t in tickers:
            key = self.cache.key("meta", t)
            info = self.cache.get_json(key)
            if info is None:
                try:
                    with _quiet_yfinance():
                        full = yf.Ticker(t).info or {}
                except Exception as exc:  # metadata is optional: record, don't fail
                    log.info("metadata unavailable for %s: %s", t, exc)
                    full = {}
                info = {k: full.get(k) for k in
                        ("shortName", "currency", "exchange", "quoteType")}
                self.cache.put_json(key, info)
            out[t] = AssetMetadata(t, info.get("shortName"), info.get("currency"),
                                   info.get("exchange"), info.get("quoteType"))
        return out
