# ✻ Equilibrium

**A Black-Litterman portfolio-construction research terminal.**

> Start with the market equilibrium. Add conviction. Quantify uncertainty.
> Construct the portfolio. Test the result.

Equilibrium downloads market data, estimates risk, reverse-optimises the market's implied
returns, blends in your views with explicit uncertainty, optimises the portfolio,
backtests it walk-forward, and stress-tests it — all from the command line, with a
polished interactive shell.

![portfolio](docs/img/portfolio.svg)

> **Research and education only. Not investment advice.** Every number on screen is tagged
> `HIST` (historical observation) or `MODEL` (model estimate); neither is a forecast.

---

## Install

One line (installs [uv](https://docs.astral.sh/uv/) if needed, then Equilibrium as an
isolated tool on Python ≥ 3.12):

```sh
curl -LsSf https://raw.githubusercontent.com/eddylam2024/equilibrium/main/install.sh | sh
```

If you already have uv:

```sh
uv tool install git+https://github.com/eddylam2024/equilibrium
```

From a clone:

```sh
git clone https://github.com/eddylam2024/equilibrium && cd equilibrium
make install          # uv tool install --editable .
```

This puts two commands on your `PATH`: `equilibrium` and its short alias `eq`.
Uninstall with `uv tool uninstall equilibrium`.

---

## Quick start

```sh
equilibrium                                   # interactive shell
equilibrium portfolio AAPL MSFT NVDA AMZN GOOGL --period 10y
equilibrium views add --long NVDA --short MSFT --spread 5% --confidence 70%
equilibrium portfolio --verbose               # reuses the saved universe; shows P, Q, Ω, M
equilibrium backtest --rebalance monthly --plot
```

No network? Add `--synthetic` to any command for a seeded, clearly labelled synthetic
market (`AAA BBB CCC DDD`, …).

### The interactive shell

Run `equilibrium` with no arguments:

* `/` opens the command palette with descriptions; `tab` completes commands, flags and
  tickers from your universe.
* Grey auto-suggestions from history (`→` to accept), `↑↓` history, `ctrl-d` to exit.
* A status bar shows the universe, number of views, `r_f`, `δ` and `τ`.
* Commands are the same as the one-shot CLI (`portfolio`, `/portfolio`, `views add …`).

Input is tokenised with `shlex` and dispatched to the command tree; nothing you type is
executed by a system shell or evaluated as Python.

---

## Commands

| Command | What it does |
|---|---|
| `data` | Download, clean and summarise adjusted prices (`--clear-cache`) |
| `analyze` | Per-asset historical return, volatility, Sharpe, Sortino, drawdown; correlation |
| `covariance` | Annualised Σ and correlation, PSD check, condition number (`--plot`) |
| `risk` | Portfolio variance/volatility, MCR/CCR risk decomposition (`--weights`) |
| `views add/list/remove/clear/omega` | Manage absolute and relative views; inspect `P`, `Q`, `Ω` |
| `portfolio` | Equilibrium → views → posterior → optimised BL portfolio vs market |
| `optimize` | `min-variance`, `max-sharpe`, `target-return`, `max-utility` on BL, Π or historical μ |
| `frontier` | Efficient frontier with min-variance, max-Sharpe, BL and market marked (`--plot`) |
| `backtest` | Walk-forward comparison of methodologies with costs (`--plot`) |
| `stress` | Historical scenarios (GFC, COVID, 2022, …, custom) and hypothetical shocks |
| `export` | JSON or CSV bundle: weights, returns, Σ, correlation, risk, views, posterior, backtest |
| `config show/init/path` | Effective settings and their origin; write `equilibrium.yaml` |
| `shell` | Interactive shell |

Every command accepts `--json` (machine-readable), `--verbose` (mathematical diagnostics),
`--period/--start/--end/--interval/--frequency`, `--rf/--delta/--tau`,
`--estimator sample|ledoit-wolf`, `--psd-policy raise|repair`,
`--long-only/--allow-short`, `--min-weight/--max-weight`. Returns and rates accept
decimals (`0.05`) or percentages (`5%`); bare numbers ≥ 1 are rejected as ambiguous.

### Examples

```sh
# Relative and absolute views with Idzorek confidence
eq views add --long NVDA --short MSFT --spread 5% --confidence 70%
eq views add --asset AAPL --return 9%            # no confidence → He-Litterman Ω = pτΣpᵀ
eq views omega                                    # P, Q, Ω, implied confidence

# Constrained optimisation
eq optimize -o target-return --target-return 15% --max-weight 0.35   # must lie in the achievable range; the error message reports it
eq optimize -o max-sharpe -r historical           # historical means (labelled HIST)

# ETFs have no market cap: supply equilibrium weights
eq portfolio SPY TLT GLD --market-weights SPY=0.6,TLT=0.3,GLD=0.1

# Stress
eq stress --scenario gfc --scenario covid
eq stress --shock NVDA=-30% --weights market

# Charts and exports
eq covariance --plot --plot-format svg
eq export --format csv --out research/
```

![backtest](docs/img/bt.svg)

---

## Methodology in brief

Full derivations, conventions and references: **[docs/methodology.md](docs/methodology.md)**.

* **Returns.** Simple returns on adjusted closes (splits and dividends), resampled by
  period-end prices. Annualised with 252/52/12.
* **Risk.** Sample or Ledoit-Wolf (2004) covariance; PSD validation with optional
  eigenvalue-clipping repair (Higham, 1988); condition-number diagnostics; Cholesky solves,
  no explicit inverses.
* **Equilibrium.** `Π = δΣw_mkt` from current market caps of the selected assets
  (Black & Litterman, 1992). Default `δ = 2.5`, `τ = 0.05` (He & Litterman, 1999).
* **Posterior.** `μ_BL = Π + τΣP'(PτΣP' + Ω)⁻¹(Q − PΠ)`, cross-checked against the canonical
  precision form; `M = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹`; predictive `Σ_post = Σ + M`.
* **Ω.** He-Litterman `pτΣp'` by default; with `--confidence C`, Idzorek (2005) in closed
  form `ω = ((1 − C)/C)·pτΣp'` — the tilt toward the view is exactly `C` of the 100% tilt.
* **Optimisation.** Markowitz (1952) mean-variance with convex max-Sharpe reformulation,
  LP feasibility checks and explicit infeasibility errors.
* **Backtest.** Walk-forward, information set = closes up to the rebalance date, drifting
  weights, costs on traded notional.

### Assumptions you should know

* The "market" is only your selected universe, weighted by **current** market caps.
* Absolute views are **total** returns and are converted to excess returns (`Q = r − r_f`).
* Views act relative to equilibrium: a view below `PΠ` is bearish even if positive.
  The CLI shows `PΠ` and the tilt for each view.
* `Ω` is diagonal (view errors independent).
* The default `r_f = 4%` is an assumption; set `--rf` or `risk_free_rate` in config.

## Data

* **Source:** Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance), isolated
  in `equilibrium/data/yahoo.py` behind the `MarketDataProvider` interface. Unofficial and
  unaudited — suitable for research, not trading.
* **Cleaning:** timezones normalised; duplicate dates removed; the sample is trimmed to the
  common window; isolated holiday gaps (≤ 5 days) forward-filled and disclosed; longer gaps,
  unknown or delisted tickers raise errors. **No security is ever substituted.**
* **Cache:** `~/.cache/equilibrium` (override `EQUILIBRIUM_CACHE_DIR`), CSV/JSON, 12h TTL for
  open-ended ranges, permanent for closed historical ranges.

## Configuration

`equilibrium config init` writes a documented `equilibrium.yaml`. Precedence:
**CLI flags > `equilibrium.yaml` > defaults**. Search order: `--config`,
`$EQUILIBRIUM_CONFIG`, `./equilibrium.yaml`, `~/.config/equilibrium/equilibrium.yaml`.

```yaml
risk_free_rate: 0.041
risk_aversion: 2.5
tau: 0.05
frequency: monthly
lookback_period: 10y
covariance_estimator: ledoit-wolf
psd_policy: raise
rebalance_frequency: monthly
backtest_lookback: 36
transaction_cost_bps: 10
constraints:
  long_only: true
  max_weight: 0.4
```

The universe and views live in an inspectable `./.equilibrium/session.json`.

## Limitations

* **No historical market caps.** Backtested market-cap and BL weights use today's share
  count × historical price (look-ahead; disclosed on every run).
* **Views in backtests** are today's views applied to the past — illustrative only.
* **Survivorship bias**: universes are chosen today.
* **Single currency assumption**: assets quoted in different currencies are not converted.
* **Frictions**: no taxes, borrowing costs, market impact or minimum lot sizes; costs are
  linear in traded notional.
* **Gaussian model**: BL and the conditional-shock stress assume normality; crises have fat
  tails and correlation spikes.
* **Yahoo Finance** availability and data quality are outside this project's control.

## Development & testing

```sh
make dev          # uv sync
make test         # 129 offline tests on synthetic and hand-computed data
make test-live    # 3 live Yahoo Finance integration tests (network)
make lint typecheck
```

The test suite includes a hand-verified three-asset Black-Litterman example (exact
fractions derived on paper), equivalence of the canonical and Woodbury posterior forms,
limiting behaviour (no views, ~0% and 100% confidence), a numerical check of the Idzorek
confidence definition, closed-form optimiser checks, look-ahead tests that rewrite future
prices, and an independent re-implementation of the backtest.

### Architecture

```
src/equilibrium/
  core/        typed vectors (excess/total, historical/model), errors
  data/        MarketDataProvider, Yahoo adapter, synthetic provider, cleaning, cache
  returns/     simple/log returns, resampling, annualisation
  risk/        covariance estimators, numerical safety, risk & performance metrics
  bl/          equilibrium, views, Ω, posterior
  optimize/    constraints, optimiser, efficient frontier
  backtest/    walk-forward engine, strategies
  stress/      historical scenarios, hypothetical shocks
  export/      JSON/CSV writers
  viz/         matplotlib charts
  config/      settings (YAML), session (universe & views)
  pipeline.py  orchestration (no presentation)
  cli/         Typer commands, Rich rendering, interactive shell
```

The quantitative packages never import the CLI.

## License

MIT. Market data is subject to its provider's terms.
