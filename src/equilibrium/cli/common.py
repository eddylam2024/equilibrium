"""Shared CLI plumbing: common options, context resolution, output modes, errors."""

from __future__ import annotations

import contextlib
import datetime as dt
import functools
import inspect
import json
import os
import sys
import typing
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.text import Text

from equilibrium.cli.theme import BULLET, ELBOW, console
from equilibrium.config.session import Session, load_session, save_session
from equilibrium.config.settings import Settings, load_settings
from equilibrium.core.errors import EquilibriumError, ValidationError
from equilibrium.core.types import Frequency, normalize_tickers
from equilibrium.data.cache import DataCache
from equilibrium.data.provider import MarketDataProvider
from equilibrium.data.synthetic import SyntheticProvider
from equilibrium.data.yahoo import YahooFinanceProvider
from equilibrium.export.writers import to_jsonable
from equilibrium.pipeline import DataWindow
from equilibrium.risk.covariance import CovarianceEstimator
from equilibrium.risk.numerics import PSDPolicy

DATA_PANEL = "Data"
MODEL_PANEL = "Model"
OUTPUT_PANEL = "Output"
CONSTRAINT_PANEL = "Constraints"


class Interval(StrEnum):
    D = "1d"
    W = "1wk"
    M = "1mo"


def parse_rate(raw: str | None, name: str, allow_ge_one: bool = False) -> float | None:
    """Parse ``0.08`` or ``8%``. Bare numbers >= 1 are ambiguous and rejected."""
    if raw is None:
        return None
    s = raw.strip()
    try:
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        value = float(s)
    except ValueError as exc:
        raise ValidationError(f"{name} {raw!r} is not a number (use 0.08 or 8%).") from exc
    if abs(value) >= 1 and not (allow_ge_one and value == 1):
        raise ValidationError(
            f"{name} {raw!r} is ambiguous: write decimals (0.08) or percentages (8%)."
        )
    return value


# ------------------------------------------------------------------ common options
@dataclass
class Common:
    period: str | None
    start: dt.datetime | None
    end: dt.datetime | None
    interval: Interval
    frequency: Frequency | None
    rf: str | None
    delta: float | None
    tau: float | None
    estimator: CovarianceEstimator | None
    psd_policy: PSDPolicy | None
    long_only: bool | None
    min_weight: float | None
    max_weight: float | None
    synthetic: bool
    no_cache: bool
    config: Path | None
    json: bool
    verbose: bool


COMMON_PARAMS: dict[str, Any] = {
    "period": Annotated[str | None, typer.Option(
        "--period", "-p", help="History length: 1y 2y 3y 5y 10y 15y 20y max.",
        rich_help_panel=DATA_PANEL)],
    "start": Annotated[dt.datetime | None, typer.Option(
        "--start", help="Start date YYYY-MM-DD (overrides --period).", formats=["%Y-%m-%d"],
        rich_help_panel=DATA_PANEL)],
    "end": Annotated[dt.datetime | None, typer.Option(
        "--end", help="End date YYYY-MM-DD.", formats=["%Y-%m-%d"],
        rich_help_panel=DATA_PANEL)],
    "interval": Annotated[Interval, typer.Option(
        "--interval", help="Download interval.", rich_help_panel=DATA_PANEL)],
    "frequency": Annotated[Frequency | None, typer.Option(
        "--frequency", "-f", help="Return frequency for estimation.",
        rich_help_panel=DATA_PANEL)],
    "synthetic": Annotated[bool, typer.Option(
        "--synthetic", help="Use seeded synthetic data (offline; NOT real prices).",
        rich_help_panel=DATA_PANEL)],
    "no_cache": Annotated[bool, typer.Option(
        "--no-cache", help="Bypass the local data cache.", rich_help_panel=DATA_PANEL)],
    "rf": Annotated[str | None, typer.Option(
        "--rf", help="Annual risk-free rate, e.g. 0.041 or 4.1%.", rich_help_panel=MODEL_PANEL)],
    "delta": Annotated[float | None, typer.Option(
        "--delta", help="Risk aversion δ (default 2.5).", rich_help_panel=MODEL_PANEL)],
    "tau": Annotated[float | None, typer.Option(
        "--tau", help="Prior uncertainty τ (default 0.05).", rich_help_panel=MODEL_PANEL)],
    "estimator": Annotated[CovarianceEstimator | None, typer.Option(
        "--estimator", help="Covariance estimator.", rich_help_panel=MODEL_PANEL)],
    "psd_policy": Annotated[PSDPolicy | None, typer.Option(
        "--psd-policy", help="If Σ is not PSD: raise, or repair by eigenvalue clipping.",
        rich_help_panel=MODEL_PANEL)],
    "long_only": Annotated[bool | None, typer.Option(
        "--long-only/--allow-short", help="Forbid or allow short positions.",
        rich_help_panel=CONSTRAINT_PANEL, show_default=False)],
    "min_weight": Annotated[float | None, typer.Option(
        "--min-weight", help="Minimum weight per asset (decimal).",
        rich_help_panel=CONSTRAINT_PANEL)],
    "max_weight": Annotated[float | None, typer.Option(
        "--max-weight", help="Maximum weight per asset (decimal).",
        rich_help_panel=CONSTRAINT_PANEL)],
    "config": Annotated[Path | None, typer.Option(
        "--config", help="Path to equilibrium.yaml.", rich_help_panel=OUTPUT_PANEL)],
    "json": Annotated[bool, typer.Option(
        "--json", help="Machine-readable JSON output.", rich_help_panel=OUTPUT_PANEL)],
    "verbose": Annotated[bool, typer.Option(
        "--verbose", "-v", help="Show mathematical diagnostics.",
        rich_help_panel=OUTPUT_PANEL)],
}
COMMON_DEFAULTS: dict[str, Any] = {k: None for k in COMMON_PARAMS} | {
    "interval": Interval.D, "synthetic": False, "no_cache": False, "json": False,
    "verbose": False,
}


def with_common(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Append the shared data/model/output options to a command's signature.

    The wrapped function receives them bundled as ``common: Common``.
    """
    hints = typing.get_type_hints(fn, include_extras=True)  # resolve string annotations
    sig = inspect.signature(fn)
    own = [p.replace(annotation=hints.get(p.name, p.annotation))
           for p in sig.parameters.values() if p.name != "common"]
    extra = [inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=COMMON_DEFAULTS[k],
                               annotation=ann) for k, ann in COMMON_PARAMS.items()]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        common = Common(**{k: kwargs.pop(k) for k in COMMON_PARAMS})
        with handle_errors(common.json):
            return fn(*args, common=common, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=own + extra)  # type: ignore[attr-defined]
    wrapper.__annotations__ = {k: v for k, v in hints.items() if k != "common"}
    wrapper.__annotations__.update(COMMON_PARAMS)
    return wrapper


# ------------------------------------------------------------------ context
@dataclass
class Ctx:
    settings: Settings
    provider: MarketDataProvider
    tickers: tuple[str, ...]
    window: DataWindow
    session: Session
    common: Common


def settings_from(common: Common) -> Settings:
    overrides = {
        "risk_free_rate": parse_rate(common.rf, "--rf"),
        "risk_aversion": common.delta,
        "tau": common.tau,
        "frequency": common.frequency,
        "covariance_estimator": common.estimator,
        "psd_policy": common.psd_policy,
        "long_only": common.long_only,
        "min_weight": common.min_weight,
        "max_weight": common.max_weight,
        "lookback_period": common.period,
    }
    settings, _ = load_settings(common.config, overrides)
    return settings


def make_provider(settings: Settings, common: Common) -> MarketDataProvider:
    if common.synthetic or os.environ.get("EQUILIBRIUM_PROVIDER") == "synthetic":
        return SyntheticProvider()
    cache = DataCache(ttl_hours=settings.cache_ttl_hours,
                      enabled=settings.cache_enabled and not common.no_cache)
    return YahooFinanceProvider(cache)


def resolve(common: Common, tickers: list[str] | None, min_assets: int = 1) -> Ctx:
    settings = settings_from(common)
    session = load_session()
    if tickers:
        syms = normalize_tickers(tickers)
        if list(syms) != session.universe:
            session.universe = list(syms)
            save_session(session)
            if not common.json:
                note(f"Universe saved: {' '.join(syms)}")
    elif session.universe:
        syms = tuple(session.universe)
    else:
        raise ValidationError(
            "No tickers given and no saved universe. Example: equilibrium portfolio AAPL MSFT NVDA"
        )
    if len(syms) < min_assets:
        raise ValidationError(f"This command needs at least {min_assets} assets.")
    window = DataWindow(
        period=common.period if not common.start else None,
        start=common.start.date() if common.start else None,
        end=common.end.date() if common.end else None,
        interval=common.interval.value,
    )
    return Ctx(settings, make_provider(settings, common), syms, window, session, common)


# ------------------------------------------------------------------ output helpers
@contextlib.contextmanager
def working(message: str, quiet: bool = False) -> Iterator[None]:
    """Animated spinner while work is in progress (silent for JSON/pipes)."""
    if quiet or not console.is_terminal:
        yield
        return
    with console.status(Text(message, style="muted"), spinner="equilibrium",
                        spinner_style="accent"):
        yield


def section(title: str, right: Text | str | None = None) -> None:
    t = Text()
    t.append(f"{BULLET} ", style="accent")
    t.append(title, style="title")
    if right is not None:
        t.append("  ")
        t.append(right if isinstance(right, Text) else Text(right, style="muted"))
    console.print()
    console.print(t)


def note(message: str | Text, style: str = "muted") -> None:
    """A ``⎿``-prefixed line whose wrapped continuation lines keep the indent."""
    from rich.table import Table

    grid = Table.grid(padding=0)
    grid.add_column(width=5, no_wrap=True)
    grid.add_column(ratio=1)
    body = message if isinstance(message, Text) else Text(message, style=style)
    grid.add_row(Text(f"  {ELBOW}  ", style="faint"), body)
    console.print(grid)


def emit_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(to_jsonable(payload), indent=2) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def handle_errors(json_mode: bool = False) -> Iterator[None]:
    try:
        yield
    except EquilibriumError as exc:
        if json_mode:
            emit_json({"error": type(exc).__name__, "message": str(exc)})
        else:
            render_error(exc)
        raise typer.Exit(1) from exc


def render_error(exc: Exception) -> None:
    kind = {
        "InvalidTickerError": "Unknown ticker", "InsufficientHistoryError": "Insufficient data",
        "InfeasibleError": "Infeasible", "NumericalError": "Numerical problem",
        "ValidationError": "Invalid input", "DataError": "Data problem",
    }.get(type(exc).__name__, "Error")
    console.print()
    console.print(Text("✗ ", style="err") + Text(kind, style="err"))
    note(str(exc), style="default")
