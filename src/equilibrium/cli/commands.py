"""Typer command definitions. Thin: parse input -> call pipeline -> render or emit JSON."""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from rich.text import Text

from equilibrium import __version__
from equilibrium.backtest.engine import BacktestConfig, BacktestResult, Rebalance, run_backtest
from equilibrium.backtest.strategies import (
    BlackLittermanStrategy,
    EqualWeight,
    MarketCap,
    MaxSharpeHistorical,
    MinVariance,
    Strategy,
)
from equilibrium.bl.views import View, ViewType
from equilibrium.cli import render
from equilibrium.cli.common import (
    OUTPUT_PANEL,
    Common,
    Ctx,
    emit_json,
    handle_errors,
    note,
    parse_rate,
    resolve,
    section,
    settings_from,
    with_common,
    working,
)
from equilibrium.cli.theme import bar, colored_pct, console, num, pct, sparkline, tag
from equilibrium.config.session import load_session, save_session, session_path
from equilibrium.config.settings import CONFIG_FILENAME, default_config_yaml, find_config
from equilibrium.core.errors import EquilibriumError, InvalidTickerError, ValidationError
from equilibrium.core.types import CovarianceMatrix, ExpectedReturns, Weights
from equilibrium.data.cache import DataCache, default_cache_dir
from equilibrium.data.provider import PriceRequest
from equilibrium.export.writers import write_csv_bundle, write_json
from equilibrium.optimize.frontier import efficient_frontier
from equilibrium.optimize.optimizer import Objective, optimize
from equilibrium.pipeline import (
    MarketSnapshot,
    asset_statistics,
    build_portfolio,
    historical_mean_returns,
    load_snapshot,
    parse_weight_spec,
    run_black_litterman,
    split_views,
)
from equilibrium.risk.metrics import (
    component_risk_contribution,
    marginal_risk_contribution,
    performance_summary,
    portfolio_variance,
    portfolio_volatility,
)
from equilibrium.stress.scenarios import (
    SCENARIOS,
    Scenario,
    historical_stress,
    hypothetical_stress,
    parse_shocks,
)

app = typer.Typer(
    name="equilibrium",
    help="[bold]✻ Equilibrium[/] — Black-Litterman portfolio construction research terminal.\n\n"
         "Start with the market equilibrium. Add conviction. Quantify uncertainty. "
         "Construct the portfolio. Test the result.\n\n"
         "[dim]Run without arguments for the interactive shell. Research and education only; "
         "not investment advice.[/]",
    rich_markup_mode="rich",
    no_args_is_help=False,
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
views_app = typer.Typer(help="Manage investor views (absolute and relative).",
                        rich_markup_mode="rich", no_args_is_help=True)
config_app = typer.Typer(help="Show or create configuration.", rich_markup_mode="rich",
                         no_args_is_help=True)
app.add_typer(views_app, name="views", rich_help_panel="Black-Litterman")
app.add_typer(config_app, name="config", rich_help_panel="Setup")

Tickers = Annotated[list[str] | None, typer.Argument(
    help="Ticker symbols. Omit to reuse the saved universe.", show_default=False)]
PlotOpt = Annotated[bool, typer.Option("--plot", help="Write PNG/SVG charts.",
                                       rich_help_panel=OUTPUT_PANEL)]
PlotDir = Annotated[Path, typer.Option("--plot-dir", help="Directory for charts.",
                                       rich_help_panel=OUTPUT_PANEL)]
PlotFmt = Annotated[str, typer.Option("--plot-format", help="png or svg.",
                                      rich_help_panel=OUTPUT_PANEL)]
MarketWeightsOpt = Annotated[str | None, typer.Option(
    "--market-weights", help="Override market-cap weights, e.g. SPY=0.6,TLT=0.4 "
                             "(needed for ETFs, which have no market cap).")]


class ReturnSource(StrEnum):
    BL = "bl"
    EQUILIBRIUM = "equilibrium"
    HISTORICAL = "historical"


def _snapshot(ctx: Ctx) -> MarketSnapshot:
    with working(f"Fetching {len(ctx.tickers)} price histories…", ctx.common.json):
        return load_snapshot(ctx.provider, ctx.tickers, ctx.settings, ctx.window)


def _meta(snap: MarketSnapshot) -> dict:
    return {
        "equilibrium_version": __version__,
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "source": snap.price_data.source,
        "synthetic": snap.synthetic,
        "start": snap.start, "end": snap.end,
        "frequency": snap.settings.frequency.value,
        "observations": len(snap.returns),
        "risk_free_rate": snap.settings.risk_free_rate,
        "notes": snap.price_data.notes,
        "disclaimer": "Research/education only. Not investment advice.",
    }


def _plot_path(directory: Path, name: str, fmt: str) -> Path:
    if fmt not in ("png", "svg"):
        raise ValidationError("--plot-format must be png or svg.")
    return directory / f"{name}.{fmt}"


def _plotted(paths: list[Path]) -> None:
    section("CHARTS")
    for p in paths:
        note(str(p))


# ================================================================== data
@app.command(rich_help_panel="Data")
@with_common
def data(tickers: Tickers = None, clear_cache: Annotated[bool, typer.Option(
        "--clear-cache", help="Delete cached downloads first.")] = False, *,
         common: Common) -> None:
    """Download, clean and summarise adjusted price history."""
    if clear_cache:
        cleared = DataCache().clear()
        note(f"Cleared {cleared} cached file(s) from {default_cache_dir()}")
    ctx = resolve(common, tickers)
    snap = _snapshot(ctx)
    px = snap.price_data.prices
    meta = None
    if not common.json:
        with working("Fetching metadata…"):
            meta = ctx.provider.get_metadata(ctx.tickers)
    if common.json:
        emit_json({"meta": _meta(snap), "prices_first": px.iloc[0], "prices_last": px.iloc[-1],
                   "observations": len(px)})
        return
    render.banner(snap, "Market data")
    section("ADJUSTED PRICES", tag("hist"))
    t = render.table(("Ticker", "left"), ("Name", "left"), ("Ccy", "left"), ("First", "right"),
                     ("Last price", "right"), ("Total return", "right"), ("", "left"))
    for tk in ctx.tickers:
        m = meta.get(tk) if meta else None
        s = px[tk]
        t.add_row(Text(tk, style="bold white"), (m.name or "—")[:28] if m else "—",
                  (m.currency or "—") if m else "—", f"{s.index[0]:%Y-%m-%d}", num(s.iloc[-1]),
                  colored_pct(s.iloc[-1] / s.iloc[0] - 1, 1), sparkline(s.to_numpy(), 28))
    console.print(render.indent(t))
    note(f"{len(px)} daily observations · adjusted for splits and dividends "
         f"({'total-return basis' if snap.price_data.adjusted else 'unadjusted'})")
    for n in snap.price_data.notes:
        note(n, style="faint")
    note(f"Cache: {default_cache_dir()}", style="faint")
    console.print()


# ================================================================== analyze
@app.command(rich_help_panel="Research")
@with_common
def analyze(tickers: Tickers = None, *, common: Common) -> None:
    """Historical return and risk statistics for each asset."""
    ctx = resolve(common, tickers)
    snap = _snapshot(ctx)
    stats = asset_statistics(snap)
    if common.json:
        emit_json({"meta": _meta(snap), "label": "historical observations",
                   "assets": {s.ticker: {**s.summary.as_dict(), "mean_return": s.mean_return}
                              for s in stats}})
        return
    render.banner(snap, "Historical analysis")
    render.data_notes(snap, common.verbose)
    section("ASSET STATISTICS", tag("hist"))
    t = render.table(("Ticker", "left"), ("Mean ret", "right"), ("CAGR", "right"),
                     ("Vol", "right"), ("Sharpe", "right"), ("Sortino", "right"),
                     ("Max DD", "right"), ("Price path", "left"))
    for s in stats:
        px = snap.price_data.prices[s.ticker]
        t.add_row(Text(s.ticker, style="bold white"), pct(s.mean_return, 1),
                  colored_pct(s.summary.cagr, 1), pct(s.summary.volatility, 1),
                  num(s.summary.sharpe), num(s.summary.sortino),
                  Text(pct(s.summary.max_drawdown, 1), style="neg"),
                  sparkline(px.to_numpy(), 30))
    console.print(render.indent(t))
    note(f"Annualised from {len(snap.returns)} {snap.settings.frequency.value} simple returns; "
         f"Sharpe/Sortino use r_f = {pct(snap.settings.risk_free_rate)}.", style="faint")
    note("Past performance describes history only; it is not an estimate of future returns.",
         style="faint")
    if len(snap.tickers) > 1:
        section("CORRELATION", tag("hist"))
        console.print(render.indent(render.matrix(snap.corr, snap.tickers, "corr")))
    console.print()


# ================================================================== covariance
@app.command(rich_help_panel="Research")
@with_common
def covariance(tickers: Tickers = None, plot: PlotOpt = False,
               plot_dir: PlotDir = Path("equilibrium-plots"), plot_format: PlotFmt = "png", *,
               common: Common) -> None:
    """Annualised covariance and correlation matrices with numerical diagnostics."""
    ctx = resolve(common, tickers, min_assets=2)
    snap = _snapshot(ctx)
    cov = snap.cov
    if common.json:
        emit_json({"meta": _meta(snap), "estimator": cov.estimator,
                   "covariance": cov.to_frame(),
                   "correlation": pd.DataFrame(snap.corr, index=snap.tickers,
                                               columns=snap.tickers),
                   "diagnostics": cov.diagnostics})
        return
    render.banner(snap, "Covariance")
    section("COVARIANCE Σ", Text(f"annualised · {cov.estimator}", style="muted"))
    console.print(render.indent(render.matrix(cov.values, snap.tickers, "cov", ".4f")))
    section("CORRELATION", tag("hist"))
    console.print(render.indent(render.matrix(snap.corr, snap.tickers, "corr")))
    d = cov.diagnostics
    section("NUMERICAL HEALTH")
    ok = Text("✓ positive semidefinite", style="pos") if d["is_psd"] else Text(
        "✗ not PSD — repaired by eigenvalue clipping", style="warn")
    rows: list[tuple[str, str | Text]] = [
        ("PSD", ok), ("Condition number", f"{float(d['condition_number']):.3g}"),
        ("Eigenvalues", f"[{float(d['min_eigenvalue']):.2e}, {float(d['max_eigenvalue']):.2e}]"),
        ("Observations", f"{int(d['observations'])} × {int(d['assets'])} assets"),
    ]
    if "shrinkage_intensity" in d:
        rows.append(("Shrinkage", num(float(d["shrinkage_intensity"]), 3)))
    console.print(render.kv(rows))
    if d.get("ill_conditioned"):
        note("Ill-conditioned: optimisation will be sensitive. Consider --estimator "
             "ledoit-wolf.", style="warn")
    if "warning" in d:
        note(str(d["warning"]), style="warn")
    if plot:
        from equilibrium.viz import charts

        cap = f"Source: {snap.price_data.source}, {snap.start} to {snap.end}. Historical."
        paths = [
            charts.heatmap(cov.values, snap.tickers, _plot_path(plot_dir, "covariance",
                           plot_format), title="Covariance (annualised)", diverging=False,
                           caption=cap, fmt=".4f"),
            charts.heatmap(snap.corr, snap.tickers, _plot_path(plot_dir, "correlation",
                           plot_format), title="Correlation", diverging=True, caption=cap,
                           fmt=".2f"),
        ]
        _plotted(paths)
    console.print()


# ================================================================== risk
@app.command(rich_help_panel="Research")
@with_common
def risk(tickers: Tickers = None,
         weights: Annotated[str | None, typer.Option(
             "--weights", "-w", help="Portfolio, e.g. AAPL=0.5,MSFT=0.5. Default: equal "
                                     "weight.")] = None,
         plot: PlotOpt = False, plot_dir: PlotDir = Path("equilibrium-plots"),
         plot_format: PlotFmt = "png", *, common: Common) -> None:
    """Portfolio risk: variance, volatility, risk contributions and historical statistics."""
    ctx = resolve(common, tickers)
    snap = _snapshot(ctx)
    spec = parse_weight_spec(weights)
    if spec:
        missing = [t for t in snap.tickers if t not in spec]
        if missing or len(spec) != len(snap.tickers):
            raise ValidationError("--weights must list every ticker in the universe exactly once.")
        w = Weights(snap.tickers, np.array([spec[t] for t in snap.tickers]))
        w.require_fully_invested(1e-6)
        label = "user-specified"
    else:
        w = Weights.equal(snap.tickers)
        label = "equal weight (default — set with --weights)"
    s = snap.cov.values
    vol = portfolio_volatility(w.values, s)
    mcr = marginal_risk_contribution(w.values, s)
    ccr = component_risk_contribution(w.values, s)
    ppy = snap.settings.frequency.periods_per_year
    realised = snap.returns.to_numpy() @ w.values
    perf = performance_summary(pd.Series(realised, index=snap.returns.index),
                               snap.settings.risk_free_rate, ppy)
    if common.json:
        emit_json({"meta": _meta(snap), "weights": w.to_series(), "weights_label": label,
                   "ex_ante": {"variance": portfolio_variance(w.values, s), "volatility": vol,
                               "mcr": dict(zip(snap.tickers, mcr, strict=True)),
                               "ccr": dict(zip(snap.tickers, ccr, strict=True))},
                   "historical": perf.as_dict()})
        return
    render.banner(snap, "Portfolio risk")
    section("RISK DECOMPOSITION", Text(label, style="muted"))
    t = render.table(("Ticker", "left"), ("Weight", "right"), ("σ asset", "right"),
                     ("MCR", "right"), ("CCR", "right"), ("% of risk", "right"), ("", "left"))
    share = ccr / vol
    for i, tk in enumerate(snap.tickers):
        t.add_row(Text(tk, style="bold white"), pct(w.values[i], 1),
                  pct(snap.cov.volatilities[i], 1), pct(mcr[i]), pct(ccr[i]), pct(share[i], 1),
                  bar(share[i], float(np.abs(share).max()), 16))
    console.print(render.indent(t))
    note("MCR = ∂σ_p/∂w_i = (Σw)_i/σ_p · CCR_i = w_i·MCR_i · ΣCCR = σ_p (Euler)", style="faint")
    section("PORTFOLIO", tag("hist"))
    console.print(render.kv([
        ("Variance  wᵀΣw", f"{portfolio_variance(w.values, s):.5f}"),
        ("Volatility  √wᵀΣw", Text(pct(vol), style="bold white")),
        ("Realised CAGR", colored_pct(perf.cagr)),
        ("Realised volatility", pct(perf.volatility)),
        ("Sharpe", num(perf.sharpe)), ("Sortino", num(perf.sortino)),
        ("Downside deviation", pct(perf.downside_deviation)),
        ("Max drawdown", Text(pct(perf.max_drawdown), style="neg")),
    ]))
    note("Realised statistics assume the weights were rebalanced every period over the "
         "sample (historical simulation).", style="faint")
    if plot:
        from equilibrium.viz import charts

        _plotted([charts.risk_contribution_chart(
            pd.Series(share, index=snap.tickers), _plot_path(plot_dir, "risk_contribution",
                                                             plot_format),
            caption=f"Ex-ante, historical Σ ({snap.start} to {snap.end}). Weights: {label}.")])
    console.print()


# ================================================================== portfolio
@app.command(rich_help_panel="Black-Litterman")
@with_common
def portfolio(tickers: Tickers = None,
              objective: Annotated[Objective, typer.Option(
                  "--objective", "-o", help="Optimisation objective on the BL posterior.")
              ] = Objective.MAX_SHARPE,
              target_return: Annotated[str | None, typer.Option(
                  "--target-return", help="Total return target for target-return, e.g. 9%.")
              ] = None,
              market_weights: MarketWeightsOpt = None,
              plot: PlotOpt = False, plot_dir: PlotDir = Path("equilibrium-plots"),
              plot_format: PlotFmt = "png", *, common: Common) -> None:
    """Equilibrium → views → posterior → optimised Black-Litterman portfolio."""
    ctx = resolve(common, tickers, min_assets=2)
    snap = _snapshot(ctx)
    with working("Reverse-optimising equilibrium and combining views…", common.json):
        bl = run_black_litterman(snap, ctx.provider, ctx.session.views,
                                 parse_weight_spec(market_weights) or ctx.session.market_weights)
        report = build_portfolio(snap, bl, objective, parse_rate(target_return,
                                                                 "--target-return"))
    if common.json:
        emit_json(portfolio_payload(report))
        return
    render.portfolio_screen(report, common.verbose)
    if plot:
        from equilibrium.viz import charts

        frame = pd.DataFrame({"Market": bl.market_weights.values,
                              "Black-Litterman": report.optimized.weights.values},
                             index=snap.tickers)
        cap = f"Model estimates. Data: {snap.price_data.source}, {snap.start} to {snap.end}."
        _plotted([
            charts.weight_comparison_chart(frame, _plot_path(plot_dir, "weights", plot_format),
                                           caption=cap),
            charts.risk_contribution_chart(
                pd.Series(report.risk_contributions, index=snap.tickers),
                _plot_path(plot_dir, "bl_risk_contribution", plot_format), caption=cap),
        ])


def portfolio_payload(report) -> dict:
    snap, bl, opt = report.snapshot, report.bl, report.optimized
    post = bl.posterior
    return {
        "meta": _meta(snap),
        "assumptions": bl.assumptions,
        "market_weights": bl.market_weights.to_series(),
        "market_weights_source": bl.market_weights_source,
        "equilibrium_excess_returns": post.prior.to_series(),
        "posterior_excess_returns": post.posterior.to_series(),
        "posterior_total_returns": post.posterior.to_total(report.rf).to_series(),
        "bl_weights": opt.weights.to_series(),
        "weight_difference": opt.weights.to_series() - bl.market_weights.to_series(),
        "objective": opt.objective.value,
        "constraints": opt.constraints.model_dump(),
        "portfolio": {"expected_excess_return": opt.expected_excess_return,
                      "expected_total_return": (opt.expected_excess_return or 0) + report.rf,
                      "volatility": opt.volatility, "sharpe": opt.sharpe,
                      "label": "model estimate"},
        "market_portfolio": {"expected_excess_return": report.market_expected_excess,
                             "volatility": report.market_volatility},
        "risk_contribution": dict(zip(snap.tickers, report.risk_contributions, strict=True)),
        "views": [v.model_dump() for v in bl.views],
        "inactive_views": [v.model_dump() for v in bl.inactive_views],
        "P": post.P, "Q": post.Q, "omega": post.omega,
        "omega_methods": [m.value for m in bl.omega_methods],
        "posterior_mean_uncertainty_M": post.mean_uncertainty,
        "predictive_covariance": post.predictive_cov.to_frame(),
        "covariance": snap.cov.to_frame(),
        "diagnostics": {**snap.cov.diagnostics, **post.diagnostics},
    }


# ================================================================== optimize
def _mu_and_cov(ctx: Ctx, snap: MarketSnapshot, source: ReturnSource
                ) -> tuple[ExpectedReturns, CovarianceMatrix, str]:
    if source is ReturnSource.HISTORICAL:
        return historical_mean_returns(snap), snap.cov, "historical sample mean"
    bl = run_black_litterman(snap, ctx.provider,
                             ctx.session.views if source is ReturnSource.BL else [],
                             ctx.session.market_weights)
    label = "BL posterior" if source is ReturnSource.BL else "equilibrium Π"
    return bl.posterior.posterior, bl.posterior.predictive_cov, label


ReturnsOpt = Annotated[ReturnSource, typer.Option(
    "--returns", "-r", help="Expected returns: bl (posterior), equilibrium (Π), or historical "
                            "(sample mean — describes the past, not a forecast).")]


@with_common
def optimize_cmd(tickers: Tickers = None,
                 objective: Annotated[Objective, typer.Option("--objective", "-o")
                                      ] = Objective.MAX_SHARPE,
                 returns: ReturnsOpt = ReturnSource.BL,
                 target_return: Annotated[str | None, typer.Option(
                     "--target-return", help="Total annual return target, e.g. 0.09 or 9%.")
                 ] = None, *, common: Common) -> None:
    """Mean-variance optimisation: min-variance, max-sharpe, target-return, max-utility."""
    ctx = resolve(common, tickers, min_assets=2)
    snap = _snapshot(ctx)
    rf = ctx.settings.risk_free_rate
    tr = parse_rate(target_return, "--target-return")
    with working("Solving…", common.json):
        mu, cov, label = _mu_and_cov(ctx, snap, returns)
        res = optimize(objective, cov, mu, ctx.settings.constraints,
                       target=None if tr is None else tr - rf,
                       risk_aversion=ctx.settings.risk_aversion)
    if common.json:
        emit_json({"meta": _meta(snap), "objective": objective.value, "returns": label,
                   "returns_source": mu.source.value, "weights": res.weights.to_series(),
                   "expected_excess_return": res.expected_excess_return,
                   "volatility": res.volatility, "sharpe": res.sharpe,
                   "constraints": res.constraints.model_dump()})
        return
    render.banner(snap, "Optimisation")
    kind = "hist" if mu.source.value == "historical" else "model"
    section(objective.value.upper(), Text(f"μ = {label} · {res.constraints.describe()}",
                                          style="muted"))
    console.print(render.indent(render.weights_table(
        snap.tickers, {"Weight": res.weights.values, "μ (excess)": mu.values})))
    section("RESULT", tag(kind))
    console.print(render.kv([
        ("Expected return (total)", Text(pct((res.expected_excess_return or 0) + rf),
                                         style="bold white")),
        ("Volatility", pct(res.volatility)), ("Sharpe (ex-ante)", num(res.sharpe)),
    ]))
    if kind == "hist":
        note("Sample means are noisy estimates of the past; optimising on them is known to "
             "produce extreme, unstable weights (Michaud, 1989).", style="warn")
    console.print()


optimize_cmd.__name__ = "optimize"
app.command(name="optimize", rich_help_panel="Black-Litterman")(optimize_cmd)


# ================================================================== frontier
@app.command(rich_help_panel="Black-Litterman")
@with_common
def frontier(tickers: Tickers = None, returns: ReturnsOpt = ReturnSource.BL,
             points: Annotated[int, typer.Option("--points", min=5, max=200)] = 40,
             plot: PlotOpt = False, plot_dir: PlotDir = Path("equilibrium-plots"),
             plot_format: PlotFmt = "png", *, common: Common) -> None:
    """Efficient frontier with min-variance, max-Sharpe, BL and market portfolios marked."""
    ctx = resolve(common, tickers, min_assets=2)
    snap = _snapshot(ctx)
    rf = ctx.settings.risk_free_rate
    with working("Tracing the efficient frontier…", common.json):
        mu, cov, label = _mu_and_cov(ctx, snap, returns)
        f = efficient_frontier(cov, mu, ctx.settings.constraints, points)
        bl = run_black_litterman(snap, ctx.provider, ctx.session.views,
                                 ctx.session.market_weights)
        bl_port = build_portfolio(snap, bl).optimized.weights.values
        m = mu.values
        marks = {"Min variance": (f.min_variance.volatility,
                                  float(f.min_variance.weights.values @ m))}
        if f.max_sharpe:
            marks["Max Sharpe"] = (f.max_sharpe.volatility, float(f.max_sharpe.weights.values @ m))
        w_m = bl.market_weights.values
        marks["Black-Litterman"] = (float(np.sqrt(bl_port @ cov.values @ bl_port)),
                                    float(bl_port @ m))
        marks["Market"] = (float(np.sqrt(w_m @ cov.values @ w_m)), float(w_m @ m))
    if common.json:
        emit_json({"meta": _meta(snap), "returns": label, "returns_source": mu.source.value,
                   "frontier": [{"volatility": v, "expected_excess_return": r}
                                for v, r in zip(f.volatilities, f.returns, strict=True)],
                   "marked": {k: {"volatility": v, "expected_excess_return": r}
                              for k, (v, r) in marks.items()}})
        return
    kind = "hist" if mu.source.value == "historical" else "model"
    render.banner(snap, "Efficient frontier")
    section("FRONTIER", Text(f"μ = {label}", style="muted"))
    t = render.table(("Volatility", "right"), ("E(r) total", "right"), ("", "left"))
    step = max(1, len(f.returns) // 10)
    hi = float(f.returns.max() + rf)
    for v, r in list(zip(f.volatilities, f.returns, strict=True))[::step]:
        t.add_row(pct(v, 1), pct(r + rf), bar(r + rf, hi, 24))
    console.print(render.indent(t))
    section("MARKED PORTFOLIOS", tag(kind))
    mt = render.table(("Portfolio", "left"), ("Volatility", "right"), ("E(r) total", "right"),
                      ("Sharpe", "right"))
    for k, (v, r) in marks.items():
        mt.add_row(Text(k, style="bold white"), pct(v, 1), pct(r + rf), num(r / v))
    console.print(render.indent(mt))
    if kind == "hist":
        note("Historical frontier: describes realised sample means, not a forecast.",
             style="warn")
    if plot:
        from equilibrium.viz import charts

        cap = (f"{'Historical sample means' if kind == 'hist' else 'Model estimates'} "
               f"({label}); r_f = {pct(rf)}. Data: {snap.start} to {snap.end}.")
        _plotted([charts.efficient_frontier_chart(
            f.volatilities, f.returns, marks, _plot_path(plot_dir, "frontier", plot_format),
            mu_label=label, caption=cap)])
    console.print()


# ================================================================== backtest
class StrategyName(StrEnum):
    ALL = "all"
    EQUAL = "equal"
    MARKET = "market"
    MINVAR = "min-variance"
    MAXSHARPE = "max-sharpe"
    BL = "bl"


@app.command(rich_help_panel="Evaluation")
@with_common
def backtest(tickers: Tickers = None,
             rebalance: Annotated[Rebalance | None, typer.Option("--rebalance")] = None,
             lookback: Annotated[int | None, typer.Option(
                 "--lookback", help="Estimation window in --frequency observations.")] = None,
             cost: Annotated[float | None, typer.Option(
                 "--cost", help="Transaction cost, bps of traded notional.")] = None,
             strategy: Annotated[list[StrategyName] | None, typer.Option(
                 "--strategy", "-s", help="Repeatable. Default: all.")] = None,
             market_weights: MarketWeightsOpt = None,
             plot: PlotOpt = False, plot_dir: PlotDir = Path("equilibrium-plots"),
             plot_format: PlotFmt = "png", *, common: Common) -> None:
    """Walk-forward backtest of alternative methodologies (no look-ahead in Σ or μ)."""
    ctx = resolve(common, tickers, min_assets=2)
    s = ctx.settings
    snap = _snapshot(ctx)
    cfg = BacktestConfig(
        rebalance=rebalance or s.rebalance_frequency,
        lookback=lookback or s.backtest_lookback, estimation_frequency=s.frequency,
        transaction_cost_bps=s.transaction_cost_bps if cost is None else cost,
        risk_free_rate=s.risk_free_rate, estimator=s.covariance_estimator,
    )
    chosen = set(strategy or [StrategyName.ALL])
    want = (lambda k: StrategyName.ALL in chosen or k in chosen)  # noqa: E731
    prices = snap.price_data.prices
    spec = parse_weight_spec(market_weights) or ctx.session.market_weights
    caps = None
    if want(StrategyName.MARKET) or want(StrategyName.BL):
        with working("Fetching market capitalisations…", common.json):
            caps = spec or ctx.provider.get_market_caps(ctx.tickers)
    strategies: list[Strategy] = []
    if want(StrategyName.EQUAL):
        strategies.append(EqualWeight())
    if want(StrategyName.MARKET) and caps:
        strategies.append(MarketCap(caps, prices.iloc[-1]))
    if want(StrategyName.MINVAR):
        strategies.append(MinVariance(s.constraints))
    if want(StrategyName.MAXSHARPE):
        strategies.append(MaxSharpeHistorical(s.constraints))
    if want(StrategyName.BL) and caps:
        active = split_views(ctx.session.views, ctx.tickers)[0]
        strategies.append(BlackLittermanStrategy(caps, prices.iloc[-1], s.constraints,
                                                 s.risk_aversion, s.tau, active))
    results: list[BacktestResult] = []
    for strat in strategies:
        with working(f"Backtesting {strat.name}…", common.json):
            results.append(run_backtest(prices, strat, cfg))
    if common.json:
        emit_json({"meta": _meta(snap), "config": cfg.__dict__, "label": "historical simulation",
                   "results": {r.strategy: {**r.summary.as_dict(),
                                            "rebalances": r.n_rebalances,
                                            "avg_turnover": r.average_turnover,
                                            "annual_turnover": r.annual_turnover,
                                            "total_costs": float(r.costs.sum()),
                                            "fallbacks_to_min_variance": r.fallbacks,
                                            "warnings": r.warnings}
                               for r in results}})
        return
    render.banner(snap, "Walk-forward backtest")
    first = min(r.returns.index[0] for r in results)
    section("CONFIGURATION")
    console.print(render.kv([
        ("Rebalance", cfg.rebalance.value), ("Estimation window",
                                             f"{cfg.lookback} {cfg.estimation_frequency.value}"),
        ("Transaction cost", f"{cfg.transaction_cost_bps:g} bps of traded notional"),
        ("Out-of-sample period", f"{first:%Y-%m-%d} → {snap.end:%Y-%m-%d}"),
        ("Constraints", s.constraints.describe()),
        ("Rebalances", ", ".join(sorted({str(r.n_rebalances) for r in results}))),
    ]))
    section("RESULTS", tag("hist"))
    note("Alternative methodologies, listed in a fixed order — not a ranking.", style="faint")
    t = render.table(("Methodology", "left"), ("Total", "right"), ("CAGR", "right"),
                     ("Vol", "right"), ("Sharpe", "right"), ("Sortino", "right"),
                     ("Max DD", "right"), ("Turn/yr", "right"), ("Growth", "left"))
    t.columns[0].no_wrap = True
    for r in results:
        su = r.summary
        t.add_row(Text(r.strategy, style="bold white"), colored_pct(su.total_return, 1),
                  colored_pct(su.cagr, 1), pct(su.volatility, 1), num(su.sharpe),
                  num(su.sortino), Text(pct(su.max_drawdown, 1), style="neg"),
                  pct(r.annual_turnover, 0),
                  sparkline((1 + r.returns).cumprod().to_numpy(), 10))
    console.print(render.indent(t))
    note("Daily simulated returns net of costs. Weights drift between rebalances. Each "
         "rebalance uses only data up to that day's close.", style="faint")
    warnings = sorted({w for r in results for w in r.warnings})
    fallbacks = [(r.strategy, r.fallbacks) for r in results if r.fallbacks]
    if warnings or fallbacks:
        section("CAVEATS")
        for w in warnings:
            note(w, style="warn")
        for name, n in fallbacks:
            note(f"{name}: {n} rebalance(s) had no positive expected excess return; "
                 "min-variance used instead.", style="warn")
    note("Survivorship bias: the universe was chosen today; assets that failed are absent.",
         style="warn")
    if plot:
        from equilibrium.viz import charts

        frame = pd.concat([r.returns for r in results], axis=1).dropna()
        cap = (f"Historical simulation, {cfg.rebalance.value} rebalancing, "
               f"{cfg.transaction_cost_bps:g} bps costs. Data: {snap.price_data.source}.")
        _plotted([charts.growth_chart(frame, _plot_path(plot_dir, "growth", plot_format),
                                      caption=cap),
                  charts.drawdown_chart(frame, _plot_path(plot_dir, "drawdown", plot_format),
                                        caption=cap)])
    console.print()


# ================================================================== stress
class WeightsChoice(StrEnum):
    BL = "bl"
    MARKET = "market"
    EQUAL = "equal"


@app.command(rich_help_panel="Evaluation")
@with_common
def stress(tickers: Tickers = None,
           scenario: Annotated[list[str] | None, typer.Option(
               "--scenario", help=f"Historical: {', '.join(SCENARIOS)} (repeatable).")] = None,
           window_start: Annotated[dt.datetime | None, typer.Option(
               "--from", formats=["%Y-%m-%d"], help="Custom historical window start.")] = None,
           window_end: Annotated[dt.datetime | None, typer.Option(
               "--to", formats=["%Y-%m-%d"], help="Custom historical window end.")] = None,
           shock: Annotated[list[str] | None, typer.Option(
               "--shock", help="Hypothetical shock, e.g. NVDA=-0.30 (repeatable).")] = None,
           weights: Annotated[str, typer.Option(
               "--weights", "-w", help="bl, market, equal, or AAPL=0.5,MSFT=0.5")] = "bl",
           *, common: Common) -> None:
    """Historical scenarios (observed) and hypothetical shocks (model estimates)."""
    ctx = resolve(common, tickers)
    snap = _snapshot(ctx)
    w = _stress_weights(ctx, snap, weights)
    scenarios: list[Scenario] = []
    for key in scenario or []:
        if key not in SCENARIOS:
            raise ValidationError(f"Unknown scenario {key!r}; choose from {', '.join(SCENARIOS)}.")
        scenarios.append(SCENARIOS[key])
    if window_start or window_end:
        if not (window_start and window_end):
            raise ValidationError("Custom windows need both --from and --to.")
        scenarios.append(Scenario("custom", "Custom window", window_start.date(),
                                  window_end.date(), "User-defined window."))
    if not scenarios and not shock:
        scenarios = [SCENARIOS["gfc"], SCENARIOS["covid"], SCENARIOS["rates2022"]]
    hist_results, hist_errors = [], []
    for sc in scenarios:
        try:
            with working(f"Replaying {sc.name}…", common.json):
                pdata = ctx.provider.get_prices(PriceRequest(
                    ctx.tickers, start=sc.start - dt.timedelta(days=10),
                    end=min(sc.end + dt.timedelta(days=3), dt.date.today()), period=None))
                hist_results.append(historical_stress(pdata.prices, w, sc))
        except InvalidTickerError as exc:  # no rows at all inside the window
            hist_errors.append((sc, f"no prices for {', '.join(exc.tickers)} in this window "
                                    "(not yet listed, or not covered). No proxy is used."))
        except EquilibriumError as exc:  # reported per scenario; the others still run
            hist_errors.append((sc, str(exc)))
    hypo = hypothetical_stress(w, snap.cov, parse_shocks(shock)) if shock else None
    if common.json:
        emit_json({"meta": _meta(snap), "weights": w.to_series(),
                   "historical": [{"scenario": r.scenario.name, "start": r.first_date,
                                   "end": r.last_date, "portfolio_return": r.portfolio_return,
                                   "max_drawdown": r.max_drawdown,
                                   "asset_returns": r.asset_returns} for r in hist_results],
                   "historical_errors": [{"scenario": s.name, "error": e}
                                         for s, e in hist_errors],
                   "hypothetical": None if hypo is None else {
                       "label": "model estimate", "shocks": hypo.shocks,
                       "direct_impact": hypo.direct_impact,
                       "propagated_impact": hypo.propagated_impact,
                       "propagated_asset_moves": hypo.propagated_asset_moves}})
        return
    render.banner(snap, "Stress testing")
    section("PORTFOLIO UNDER TEST")
    console.print(render.indent(render.weights_table(snap.tickers, {"Weight": w.values})))
    if hist_results or hist_errors:
        section("HISTORICAL SCENARIOS", tag("hist"))
        t = render.table(("Scenario", "left"), ("Window", "left"), ("Portfolio", "right"),
                         ("Max DD", "right"), ("Worst asset", "left"))
        for r in hist_results:
            worst = str(r.asset_returns.idxmin())
            t.add_row(Text(r.scenario.name, style="bold white"),
                      f"{r.first_date:%Y-%m-%d} → {r.last_date:%Y-%m-%d}",
                      colored_pct(r.portfolio_return, 1), Text(pct(r.max_drawdown, 1),
                                                                style="neg"),
                      f"{worst} {pct(r.asset_returns[worst], 1)}")
        console.print(render.indent(t))
        note("Buy-and-hold with today's weights over the observed window.", style="faint")
        for sc, err in hist_errors:
            note(f"{sc.name}: {err}", style="warn")
    if hypo:
        section("HYPOTHETICAL SHOCK", tag("model"))
        t = render.table(("Ticker", "left"), ("Weight", "right"), ("Direct", "right"),
                         ("Via Σ (conditional)", "right"))
        for i, tk in enumerate(snap.tickers):
            t.add_row(Text(tk, style="bold white"), pct(w.values[i], 1),
                      pct(hypo.direct_asset_moves[tk], 1, sign=True),
                      pct(hypo.propagated_asset_moves[tk], 1, sign=True))
        t.add_row(Text("Portfolio", style="accent.b"), "", colored_pct(hypo.direct_impact),
                  colored_pct(hypo.propagated_impact))
        console.print(render.indent(t))
        note("Direct: only shocked assets move. Via Σ: others move by E[r_U | r_K = s] = "
             "Σ_UK Σ_KK⁻¹ s (zero-mean Gaussian approximation; crises have fatter tails "
             "and higher correlations).", style="faint")
    console.print()


def _stress_weights(ctx: Ctx, snap: MarketSnapshot, choice: str) -> Weights:
    if "=" in choice:
        spec = parse_weight_spec(choice) or {}
        if set(spec) != set(snap.tickers):
            raise ValidationError("--weights must cover exactly the universe.")
        w = Weights(snap.tickers, np.array([spec[t] for t in snap.tickers]))
        w.require_fully_invested(1e-6)
        return w
    try:
        kind = WeightsChoice(choice)
    except ValueError as exc:
        raise ValidationError("--weights must be bl, market, equal, or TICKER=w,...") from exc
    if kind is WeightsChoice.EQUAL:
        return Weights.equal(snap.tickers)
    bl = run_black_litterman(snap, ctx.provider, ctx.session.views, ctx.session.market_weights)
    if kind is WeightsChoice.MARKET:
        return bl.market_weights
    return build_portfolio(snap, bl).optimized.weights


# ================================================================== export
class ExportFormat(StrEnum):
    JSON = "json"
    CSV = "csv"


@app.command(rich_help_panel="Output")
@with_common
def export(tickers: Tickers = None,
           fmt: Annotated[ExportFormat, typer.Option("--format", help="json or csv.")
                          ] = ExportFormat.JSON,
           out: Annotated[Path, typer.Option("--out", help="File (json) or directory (csv).")
                          ] = Path("equilibrium-export"),
           with_backtest: Annotated[bool, typer.Option("--backtest/--no-backtest")] = True,
           *, common: Common) -> None:
    """Export weights, returns, Σ, correlation, risk, views, posterior and backtests."""
    ctx = resolve(common, tickers, min_assets=2)
    s = ctx.settings
    snap = _snapshot(ctx)
    with working("Building research bundle…", common.json):
        bl = run_black_litterman(snap, ctx.provider, ctx.session.views,
                                 ctx.session.market_weights)
        report = build_portfolio(snap, bl)
        stats = asset_statistics(snap)
        bt_frame = None
        if with_backtest:
            caps = ctx.session.market_weights or ctx.provider.get_market_caps(ctx.tickers)
            px = snap.price_data.prices
            cfg = BacktestConfig(rebalance=s.rebalance_frequency, lookback=s.backtest_lookback,
                                 estimation_frequency=s.frequency,
                                 transaction_cost_bps=s.transaction_cost_bps,
                                 risk_free_rate=s.risk_free_rate,
                                 estimator=s.covariance_estimator)
            strategies: list[Strategy] = [
                EqualWeight(), MarketCap(caps, px.iloc[-1]), MinVariance(s.constraints),
                MaxSharpeHistorical(s.constraints),
                BlackLittermanStrategy(caps, px.iloc[-1], s.constraints, s.risk_aversion,
                                       s.tau, split_views(ctx.session.views, ctx.tickers)[0])]
            bts = [run_backtest(px, st, cfg) for st in strategies]
            bt_frame = pd.DataFrame({r.strategy: {**r.summary.as_dict(),
                                                  "rebalances": r.n_rebalances,
                                                  "annual_turnover": r.annual_turnover}
                                     for r in bts}).T
    payload = portfolio_payload(report)
    payload["asset_statistics"] = {st.ticker: {**st.summary.as_dict(),
                                               "mean_return": st.mean_return} for st in stats}
    payload["correlation"] = pd.DataFrame(snap.corr, index=snap.tickers, columns=snap.tickers)
    if bt_frame is not None:
        payload["backtest"] = bt_frame
    if fmt is ExportFormat.JSON:
        path = out if out.suffix == ".json" else out.with_suffix(".json")
        written = [write_json(payload, path)]
    else:
        tables = {
            "weights": pd.DataFrame({"market": bl.market_weights.to_series(),
                                     "black_litterman": report.optimized.weights.to_series(),
                                     "difference": payload["weight_difference"]}),
            "expected_returns": pd.DataFrame({
                "equilibrium_excess": bl.posterior.prior.to_series(),
                "posterior_excess": bl.posterior.posterior.to_series(),
                "posterior_total": payload["posterior_total_returns"],
                "historical_mean_total": pd.Series({st.ticker: st.mean_return for st in stats}),
            }),
            "covariance": snap.cov.to_frame(),
            "correlation": payload["correlation"],
            "posterior_predictive_covariance": bl.posterior.predictive_cov.to_frame(),
            "risk_metrics": pd.DataFrame({st.ticker: st.summary.as_dict() for st in stats}).T,
            "views": pd.DataFrame([{**v.model_dump(), "omega": bl.posterior.omega[k, k],
                                    "omega_method": bl.omega_methods[k].value}
                                   for k, v in enumerate(bl.views)]),
        }
        if bt_frame is not None:
            tables["backtest"] = bt_frame
        meta = {**_meta(snap), "assumptions": bl.assumptions,
                "portfolio": payload["portfolio"], "diagnostics": payload["diagnostics"]}
        written = write_csv_bundle(tables, meta, out)
    if common.json:
        emit_json({"written": [str(p) for p in written]})
        return
    section("EXPORTED", Text(f"{len(written)} file(s)", style="muted"))
    for p in written:
        note(str(p))
    if snap.synthetic:
        note("Contains SYNTHETIC data (flagged in metadata).", style="warn")
    console.print()


# ================================================================== views
def _print_views() -> None:
    session = load_session()
    path = session_path()
    shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    section("INVESTOR VIEWS", f"{len(session.views)} saved · {shown}")
    if not session.views:
        note("No views yet. Examples:")
        note("equilibrium views add --asset NVDA --return 8%", style="faint")
        note("equilibrium views add --long NVDA --short MSFT --spread 5% --confidence 70%",
             style="faint")
    else:
        console.print(render.indent(render.views_table(session.views)))
        note("Absolute views are total returns; converted to excess (minus r_f) in the model.",
             style="faint")
    console.print()


@views_app.command("add")
def views_add(
    asset: Annotated[str | None, typer.Option("--asset", "-a", help="Absolute view asset.")
                     ] = None,
    ret: Annotated[str | None, typer.Option("--return", help="Absolute view: expected total "
                                                              "annual return (8% or 0.08).")
                   ] = None,
    long: Annotated[str | None, typer.Option("--long", help="Relative view: outperformer.")
                    ] = None,
    short: Annotated[str | None, typer.Option("--short", help="Relative view: underperformer.")
                     ] = None,
    spread: Annotated[str | None, typer.Option("--spread", help="Relative view: annual "
                                                                "outperformance (5% or 0.05).")
                      ] = None,
    confidence: Annotated[str | None, typer.Option(
        "--confidence", "-c", help="Idzorek confidence in (0, 100%]. Omit for He-Litterman "
                                   "Ω = pτΣpᵀ (equivalent to 50%).")] = None,
    omega: Annotated[float | None, typer.Option("--omega", help="Explicit view variance ω.")
                     ] = None,
) -> None:
    """Add a view. Absolute: --asset --return. Relative: --long --short --spread."""
    with handle_errors():
        conf = parse_rate(confidence, "--confidence", allow_ge_one=True)
        view = _make_view(asset, ret, long, short, spread, conf, omega)
        session = load_session()
        if any(v.type == view.type and v.assets == view.assets for v in session.views):
            raise ValidationError(
                f"A view on {' vs '.join(view.assets)} already exists. Remove it first "
                "(equilibrium views remove N) — duplicate views would double-count conviction.")
        if session.universe:
            missing = [a for a in view.assets if a not in session.universe]
            if missing:
                note(f"{', '.join(missing)} not in current universe "
                     f"({' '.join(session.universe)}); it must be included before use.",
                     style="warn")
        session.views.append(view)
        save_session(session)
        console.print(Text("  ✓ ", style="pos") + Text("Added ") +
                      render.view_label(view) + Text(f"  {view.describe()}", style="muted"))
        _print_views()


def _make_view(asset: str | None, ret: str | None, long: str | None, short: str | None,
               spread: str | None, conf: float | None, omega: float | None) -> View:
    from pydantic import ValidationError as PydanticError

    try:
        if asset and ret is not None and not (long or short or spread):
            return View(type=ViewType.ABSOLUTE, asset=asset,
                        value=parse_rate(ret, "--return") or 0.0, confidence=conf, omega=omega)
        if long and short and spread is not None and not (asset or ret):
            return View(type=ViewType.RELATIVE, long=long, short=short,
                        value=parse_rate(spread, "--spread") or 0.0, confidence=conf,
                        omega=omega)
    except PydanticError as exc:
        raise ValidationError("; ".join(str(e["msg"]).removeprefix("Value error, ")
                                        for e in exc.errors())) from exc
    raise ValidationError("Use either --asset X --return R, or --long X --short Y --spread S.")


@views_app.command("list")
def views_list() -> None:
    """List saved views."""
    with handle_errors():
        _print_views()


@views_app.command("remove")
def views_remove(index: Annotated[int, typer.Argument(help="View number from 'views list'.")]
                 ) -> None:
    """Remove a view by number."""
    with handle_errors():
        session = load_session()
        if not 1 <= index <= len(session.views):
            raise ValidationError(f"No view #{index}; there are {len(session.views)}.")
        removed = session.views.pop(index - 1)
        save_session(session)
        console.print(Text("  ✓ Removed ", style="pos") + Text(removed.describe()))
        _print_views()


@views_app.command("clear")
def views_clear() -> None:
    """Remove all views."""
    with handle_errors():
        session = load_session()
        n = len(session.views)
        session.views = []
        save_session(session)
        console.print(Text(f"  ✓ Cleared {n} view(s)", style="pos"))


@views_app.command("omega")
@with_common
def views_omega(tickers: Tickers = None, *, common: Common) -> None:
    """Inspect P, Q and Ω for the saved views on current data."""
    ctx = resolve(common, tickers, min_assets=2)
    snap = _snapshot(ctx)
    bl = run_black_litterman(snap, ctx.provider, ctx.session.views, ctx.session.market_weights)
    if common.json:
        post = bl.posterior
        emit_json({"P": post.P, "Q": post.Q, "omega": post.omega,
                   "methods": [m.value for m in bl.omega_methods], "tickers": snap.tickers})
        return
    render.banner(snap, "View uncertainty")
    if not bl.views:
        note("No views saved.")
        return
    render.verbose_bl(snap, bl)
    note("Idzorek: ω = ((1−C)/C)·pτΣpᵀ. He-Litterman: ω = pτΣpᵀ. Ω is diagonal "
         "(independent view errors).", style="faint")
    console.print()


# ================================================================== config
@config_app.command("show")
def config_show(config: Annotated[Path | None, typer.Option("--config")] = None) -> None:
    """Show effective settings and where they come from."""
    from equilibrium.cli.common import COMMON_DEFAULTS, Common

    with handle_errors():
        common = Common(**{**COMMON_DEFAULTS, "config": config})
        s = settings_from(common)
        found = find_config(config)
        section("CONFIGURATION", str(found) if found else "defaults (no equilibrium.yaml found)")
        t = render.table(("Setting", "left"), ("Value", "right"), ("Origin", "left"))
        dump = s.model_dump()
        for key, value in dump.items():
            origin = "file" if key in s.model_fields_set else "default"
            shown = value.value if hasattr(value, "value") else value
            if isinstance(value, dict):
                shown = ", ".join(f"{k}={v}" for k, v in value.items())
            t.add_row(key, str(shown), Text(origin, style="accent" if origin == "file"
                                            else "muted"))
        console.print(render.indent(t))
        note(f"Session (universe & views): {session_path()}", style="faint")
        note(f"Data cache: {default_cache_dir()}", style="faint")
        console.print()


@config_app.command("init")
def config_init(force: Annotated[bool, typer.Option("--force")] = False) -> None:
    """Write a documented equilibrium.yaml in the current directory."""
    with handle_errors():
        path = Path.cwd() / CONFIG_FILENAME
        if path.exists() and not force:
            raise ValidationError(f"{path} exists; use --force to overwrite.")
        path.write_text(default_config_yaml())
        console.print(Text("  ✓ Wrote ", style="pos") + Text(str(path)))


@config_app.command("path")
def config_path() -> None:
    """Print the config, session and cache locations."""
    console.print(render.kv([("Config", str(find_config() or "none")),
                             ("Session", str(session_path())),
                             ("Cache", str(default_cache_dir()))]))


# ================================================================== misc
@app.command(rich_help_panel="Setup")
def shell() -> None:
    """Interactive research shell (also: run `equilibrium` with no arguments)."""
    from equilibrium.cli.shell import run_shell

    run_shell()


@app.command(rich_help_panel="Setup")
def version() -> None:
    """Print the version."""
    console.print(f"equilibrium {__version__}")
