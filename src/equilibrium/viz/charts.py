"""Publication-quality static charts (PNG/SVG) with matplotlib.

Colour follows a validated colour-vision-deficiency-safe palette: fixed-order categorical
hues, a single-hue sequential ramp for magnitude (covariance), and a blue/red diverging
ramp with a neutral grey midpoint for correlation. Every chart states whether its values
are historical observations or model estimates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from equilibrium.risk.metrics import drawdown_series

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
          "#e34948"]
SEQ = LinearSegmentedColormap.from_list("eq_seq", ["#f4f8fe", "#9ec5f4", "#3987e5", "#1c5cab",
                                                   "#0d366b"])
DIV = LinearSegmentedColormap.from_list("eq_div", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89",
                                                   "#b42c2b"])


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.labelsize": 9.5, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
        "font.family": "sans-serif", "lines.linewidth": 2.0, "figure.dpi": 110,
    })


def _finish(fig, ax, path: Path, caption: str) -> Path:
    fig.text(0.01, 0.01, caption, fontsize=7.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _pct(ax, axis: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter

    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(PercentFormatter(1.0, 0))


def heatmap(matrix: np.ndarray, labels: Sequence[str], path: Path, *, title: str,
            diverging: bool, caption: str, fmt: str) -> Path:
    _style()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.4, 0.95 * n + 1.6))
    ax.grid(False)
    if diverging:
        im = ax.imshow(matrix, cmap=DIV, vmin=-1, vmax=1)
    else:
        im = ax.imshow(matrix, cmap=SEQ, vmin=0 if matrix.min() >= 0 else None)
    ax.set_xticks(range(n), labels, rotation=0)
    ax.set_yticks(range(n), labels)
    for spine in ax.spines.values():
        spine.set_visible(False)
    lo, hi = (-1.0, 1.0) if diverging else (float(matrix.min()), float(matrix.max()))
    for i in range(n):
        for j in range(n):
            shade = (float(matrix[i, j]) - lo) / (hi - lo) if hi > lo else 0.5
            dark = (abs(shade - 0.5) > 0.3) if diverging else shade > 0.55
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=8,
                    color="white" if dark else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    ax.set_title(title)
    return _finish(fig, ax, path, caption)


def efficient_frontier_chart(vols: np.ndarray, rets: np.ndarray, points: dict[str, tuple[float,
                             float]], path: Path, *, mu_label: str, caption: str) -> Path:
    """Frontier line plus marked portfolios. ``rets`` are expected excess returns."""
    _style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(vols, rets, color=SERIES[0], lw=2, label="Efficient frontier", zorder=2)
    markers = ["o", "D", "s", "^", "v", "P"]
    merged: dict[str, tuple[float, float]] = {}
    for name, (v, r) in points.items():  # coincident portfolios share one mark and label
        hit = next((k for k, (v2, r2) in merged.items()
                    if abs(v - v2) < 1e-4 and abs(r - r2) < 1e-4), None)
        if hit is None:
            merged[name] = (v, r)
        else:
            merged[f"{hit} = {name}"] = merged.pop(hit)
    for k, (name, (v, r)) in enumerate(merged.items()):
        ax.scatter([v], [r], s=70, marker=markers[k % len(markers)],
                   color=SERIES[(k + 1) % len(SERIES)], edgecolor=SURFACE, linewidth=2,
                   zorder=4, label=name)
        ax.annotate(name, (v, r), xytext=(8, -3), textcoords="offset points", fontsize=8.5,
                    color=INK)
    ax.set_xlabel("Volatility (annualised)")
    ax.set_ylabel(f"Expected excess return — {mu_label}")
    _pct(ax, "x")
    _pct(ax, "y")
    ax.legend(loc="lower right")
    ax.set_title("Efficient frontier")
    return _finish(fig, ax, path, caption)


def growth_chart(returns: pd.DataFrame, path: Path, *, caption: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    wealth: pd.DataFrame = (returns + 1.0).cumprod()
    for k, col in enumerate(wealth.columns):
        ax.plot(wealth.index, wealth[col], color=SERIES[k % len(SERIES)], lw=1.8, label=col)
        ax.annotate(f"{wealth[col].iloc[-1]:.2f}×", (wealth.index[-1], wealth[col].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK_2,
                    va="center")
    ax.set_ylabel("Growth of $1 (historical simulation)")
    ax.legend(loc="upper left")
    ax.set_title("Portfolio growth")
    return _finish(fig, ax, path, caption)


def drawdown_chart(returns: pd.DataFrame, path: Path, *, caption: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, col in enumerate(returns.columns):
        dd = drawdown_series(returns[col])
        ax.plot(dd.index, dd, color=SERIES[k % len(SERIES)], lw=1.4, label=col)
    _pct(ax)
    ax.set_ylabel("Drawdown from peak")
    ax.legend(loc="lower left")
    ax.set_title("Drawdowns")
    return _finish(fig, ax, path, caption)


def weight_comparison_chart(weights: pd.DataFrame, path: Path, *, caption: str) -> Path:
    """Grouped horizontal bars: one group per ticker, one bar per portfolio."""
    _style()
    n, k = weights.shape
    fig, ax = plt.subplots(figsize=(8, 0.55 * n * k + 1.6))
    height = 0.8 / k
    y = np.arange(n)
    for j, col in enumerate(weights.columns):
        ax.barh(y + (j - (k - 1) / 2) * height, weights[col], height=height * 0.9,
                color=SERIES[j % len(SERIES)], label=col, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(y, weights.index)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.8)
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Portfolio weights")
    return _finish(fig, ax, path, caption)


def risk_contribution_chart(contrib: pd.Series, path: Path, *, caption: str) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(contrib) + 1.6))
    values = contrib.to_numpy(dtype=float)
    ax.barh(list(contrib.index), values, color=SERIES[0], height=0.6)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1%}", (v, i), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.8)
    _pct(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of portfolio volatility (Euler decomposition)")
    ax.set_title("Risk contribution")
    return _finish(fig, ax, path, caption)
