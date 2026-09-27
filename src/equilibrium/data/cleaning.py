"""Data cleaning. Every modification is recorded as a human-readable note.

Policy
------
* Timestamps are normalised to timezone-naive calendar dates (exchange-local trading date).
* Duplicate dates keep the last observation.
* Columns with no data at all -> InvalidTickerError (never silently dropped or substituted).
* Different trading calendars / late listings: the sample is restricted to the common
  window where every asset has a price; interior gaps of up to ``max_ffill`` days (holidays
  on one exchange but not another) are forward-filled, which yields a zero return on that
  day. Longer gaps raise an error instead of fabricating prices.
* Non-positive prices are rejected.
"""

from __future__ import annotations

import pandas as pd

from equilibrium.core.errors import DataError, InsufficientHistoryError, InvalidTickerError

MAX_FFILL_DAYS = 5


def clean_prices(raw: pd.DataFrame, tickers: tuple[str, ...], min_observations: int = 3,
                 max_ffill: int = MAX_FFILL_DAYS) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    df = raw.copy()
    missing_cols = [t for t in tickers if t not in df.columns]
    empty = [t for t in tickers if t in df.columns and df[t].dropna().empty]
    if missing_cols or empty:
        raise InvalidTickerError(missing_cols + empty,
                                 "They may be mistyped, delisted, or not covered by the source.")
    df = df[list(tickers)]

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df.sort_index()
    dupes = int(df.index.duplicated(keep="last").sum())
    if dupes:
        df = df[~df.index.duplicated(keep="last")]
        notes.append(f"Removed {dupes} duplicate timestamp(s), keeping the last value.")

    df = df.apply(pd.to_numeric, errors="coerce")
    if (df <= 0).any().any():
        bad = df.columns[(df <= 0).any()].tolist()
        raise DataError(f"Non-positive prices for {', '.join(bad)}; data is corrupt.")

    first_valid = df.apply(lambda s: s.first_valid_index())
    last_valid = df.apply(lambda s: s.last_valid_index())
    start, end = first_valid.max(), last_valid.min()
    late = [t for t in tickers if first_valid[t] > df.index[0]]
    if late:
        notes.append(
            "Sample starts " + start.date().isoformat() + " (common history begins when "
            + ", ".join(f"{t} ({first_valid[t].date()})" for t in late) + " start trading)."
        )
    early_end = [t for t in tickers if last_valid[t] < df.index[-1]]
    if early_end:
        notes.append("Data ends early for " + ", ".join(
            f"{t} ({last_valid[t].date()})" for t in early_end) + " — possibly delisted.")
    df = df.loc[start:end]

    rows_before = len(df)
    all_missing_rows = df.isna().all(axis=1)
    df = df[~all_missing_rows]
    gaps = df.isna()
    if gaps.any().any():
        run_lengths = gaps.apply(lambda s: s.groupby((~s).cumsum()).sum().max())
        too_long = run_lengths[run_lengths > max_ffill]
        if not too_long.empty:
            raise DataError(
                "Gaps longer than "
                f"{max_ffill} observations for: "
                + ", ".join(f"{t} ({int(n)})" for t, n in too_long.items())
                + ". Refusing to fabricate prices."
            )
        filled = int(gaps.sum().sum())
        df = df.ffill()
        notes.append(
            f"Forward-filled {filled} isolated missing price(s) (non-overlapping trading "
            "calendars / holidays); each yields a zero return on that date."
        )
    if rows_before - len(df):
        notes.append(f"Dropped {rows_before - len(df)} date(s) with no prices for any asset.")

    if len(df) < min_observations:
        raise InsufficientHistoryError(
            f"Only {len(df)} common observations across {', '.join(tickers)}; "
            f"at least {min_observations} are required."
        )
    return df, notes
