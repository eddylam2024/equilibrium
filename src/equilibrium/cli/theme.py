"""Visual language of the terminal UI: palette, glyphs, number formatting, small widgets."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

import numpy as np
from rich.console import Console
from rich.spinner import SPINNERS
from rich.text import Text
from rich.theme import Theme

from equilibrium import __version__

ACCENT = "#d97757"
POS = "#4eba65"
NEG = "#e5534b"
HIST = "#6cb6ff"
MODEL = "#b392f0"
MUTED = "grey50"
FAINT = "grey35"

THEME = Theme({
    "accent": ACCENT,
    "accent.b": f"bold {ACCENT}",
    "pos": POS,
    "neg": NEG,
    "muted": MUTED,
    "faint": FAINT,
    "hist": HIST,
    "model": MODEL,
    "title": "bold white",
    "label": "grey70",
    "warn": "#e3b341",
    "err": f"bold {NEG}",
})

# Breathing-asterisk spinner.
SPINNERS["equilibrium"] = {"interval": 110, "frames": ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶",
                                                        "✳", "✢"]}

BULLET = "⏺"
ELBOW = "⎿"
STAR = "✻"
SPARK = "▁▂▃▄▅▆▇█"


def make_console() -> Console:
    no_color = bool(os.environ.get("NO_COLOR"))
    return Console(theme=THEME, highlight=False, no_color=no_color, soft_wrap=False)


console = make_console()


def is_interactive() -> bool:
    return console.is_terminal and sys.stdin.isatty()


# ------------------------------------------------------------------ number formatting
def pct(x: float | None, digits: int = 2, sign: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{digits}%}" if sign else f"{x:.{digits}%}"


def num(x: float | None, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def signed(x: float, digits: int = 2, suffix: str = "") -> Text:
    """Coloured delta with ▲/▼, e.g. for BL minus market weight (in percentage points)."""
    if abs(x) < 0.5 * 10 ** (-digits):
        return Text(f"  {0:.{digits}f}{suffix}", style="muted")
    arrow, style = ("▲", "pos") if x > 0 else ("▼", "neg")
    return Text(f"{arrow} {abs(x):.{digits}f}{suffix}", style=style)


def colored_pct(x: float, digits: int = 2) -> Text:
    return Text(pct(x, digits), style="pos" if x > 0 else "neg" if x < 0 else "muted")


def bar(value: float, max_abs: float, width: int = 16, style: str = ACCENT) -> Text:
    """Horizontal bar with eighth-block resolution; negatives drawn in the negative colour."""
    if max_abs <= 0 or not np.isfinite(value):
        return Text("")
    cells = abs(value) / max_abs * width
    full = int(cells)
    rem = int((cells - full) * 8)
    s = "█" * full + ("▏▎▍▌▋▊▉"[rem - 1] if rem else "")
    return Text(s, style=NEG if value < 0 else style)


def sparkline(values: Sequence[float] | np.ndarray, width: int = 24) -> Text:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return Text("")
    if v.size > width:
        idx = np.linspace(0, v.size - 1, width).round().astype(int)
        v = v[idx]
    lo, hi = v.min(), v.max()
    scaled = np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)
    chars = "".join(SPARK[min(int(s * (len(SPARK) - 1) + 0.5), len(SPARK) - 1)] for s in scaled)
    return Text(chars, style=POS if v[-1] >= v[0] else NEG)


def tag(kind: str) -> Text:
    """Small label marking a number as a historical observation or a model estimate."""
    if kind == "hist":
        return Text(" HIST ", style=f"bold black on {HIST}")
    if kind == "model":
        return Text(" MODEL ", style=f"bold black on {MODEL}")
    return Text(f" {kind.upper()} ", style="bold black on grey62")


def _mix(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> str:
    r, g, b = (round(a + (b_ - a) * t) for a, b_ in zip(c1, c2, strict=True))
    return f"#{r:02x}{g:02x}{b:02x}"


def diverging_cell(value: float, text: str) -> Text:
    """Background-shaded cell: blue (−1) · neutral (0) · red (+1)."""
    neutral, pos, neg = (48, 48, 46), (170, 52, 50), (38, 98, 180)
    t = min(abs(value), 1.0)
    bg = _mix(neutral, pos if value >= 0 else neg, t)
    return Text(f" {text} ", style=f"white on {bg}")


def sequential_cell(value: float, lo: float, hi: float, text: str) -> Text:
    t = 0.0 if hi == lo else (value - lo) / (hi - lo)
    return Text(f" {text} ", style=f"white on {_mix((40, 44, 52), (28, 92, 171), t)}")


def header(subtitle: str = "Black-Litterman Portfolio Construction") -> Text:
    t = Text()
    t.append(f"{STAR} ", style="accent.b")
    t.append("EQUILIBRIUM", style="bold white")
    t.append(f"  {subtitle}", style="muted")
    t.append(f"   v{__version__} · research only", style="faint")
    return t
