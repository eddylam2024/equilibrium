# Methodology

This document specifies every quantitative method implemented in Equilibrium, the
conventions it uses, and the assumptions behind each default. Equations are written in
plain notation; vectors are columns, `'` denotes transpose.

> Equilibrium is a research and education tool. Nothing here is investment advice.

---

## 1. Notation and conventions

| Symbol | Meaning |
|---|---|
| `N` | number of assets; `K` number of views |
| `w` | portfolio weights, an `N`-vector aligned to an explicit ticker order, `1'w = 1` |
| `R_t` | simple (arithmetic) return `P_t / P_{t-1} - 1` |
| `r_f` | annual risk-free rate (effective); periodic `r_f,p = (1 + r_f)^(1/k) - 1` |
| `k` | periods per year: 252 (daily), 52 (weekly), 12 (monthly) |
| `Σ` | annualised covariance matrix of returns (never the correlation matrix) |
| `Π` | implied equilibrium **excess** returns |
| `μ_BL` | Black-Litterman posterior mean **excess** returns |

**Excess vs total.** Every return vector in the code carries a tag: `EXCESS` or `TOTAL`,
and `HISTORICAL` or `MODEL`. The optimiser refuses total returns, the posterior refuses a
total-return prior, and the CLI labels every number as `HIST` or `MODEL`. This prevents
the most common Black-Litterman implementation error: subtracting `r_f` twice, or not at all.

**Prices.** Adjusted closes (splits and cash dividends) from Yahoo Finance, so percentage
changes approximate total returns. Lower frequencies are formed by sampling the last price
in each week/month (keeping the actual observation date) and differencing, never by summing
daily simple returns.

**Simple vs log returns.** Simple returns aggregate exactly across assets
(`R_p = w'R`), which is what portfolio construction needs; log returns aggregate exactly
through time but not across assets. Simple returns are the default everywhere.

**Annualisation.** Means scale by `k`, covariances by `k`, volatilities by `√k`. This is
exact for i.i.d. log returns and the standard approximation for simple returns.

---

## 2. Mean-variance optimisation (Markowitz, 1952)

Markowitz (1952) formalised portfolio selection as a trade-off between expected return and
variance, showing that diversification depends on covariances and defining the efficient
set of portfolios that minimise variance for a given expected return.

Portfolio variance and volatility:

```
σ_p² = w'Σw          σ_p = √(w'Σw)
```

Risk decomposition (Euler allocation, since `σ_p` is homogeneous of degree one in `w`):

```
MCR_i = ∂σ_p/∂w_i = (Σw)_i / σ_p          (marginal contribution)
CCR_i = w_i · MCR_i,   Σ_i CCR_i = σ_p     (component contribution)
```

### Objectives

All portfolios are fully invested (`1'w = 1`), subject to box constraints
`lo ≤ w_i ≤ hi` derived from `--long-only`, `--min-weight` and `--max-weight`.

| Objective | Problem |
|---|---|
| `min-variance` | `min w'Σw` |
| `target-return` | `min w'Σw` s.t. `w'μ = target` |
| `max-sharpe` | `max w'μₓ / √(w'Σw)` (`μₓ` = excess returns) |
| `max-utility` | `max w'μₓ − (δ/2) w'Σw` |

**Max-Sharpe** is not convex in `w`. It is solved through the standard homogenisation
(Cornuéjols & Tütüncü, 2007): with `y = κw`,

```
min y'Σy   s.t.   μₓ'y = 1,   1'y = κ,   κ ≥ 0,   lo·κ ≤ y_i ≤ hi·κ
```

which is a convex QP; `w = y/κ`. If no feasible portfolio has a positive expected excess
return the tangency portfolio does not exist and Equilibrium says so instead of returning
something arbitrary.

**Feasibility** is checked before optimising: `N·lo ≤ 1 ≤ N·hi` for the budget constraint,
and a pair of linear programmes computes the achievable range `[min w'μ, max w'μ]` before a
target-return problem is attempted. Infeasible requests produce an explicit error with the
achievable range.

**Validation.** Tests compare the numerical solutions with closed forms in the
unconstrained case: `w_mv = Σ⁻¹1 / 1'Σ⁻¹1` and `w_tan ∝ Σ⁻¹μₓ`.

**A caveat.** Optimising on sample means produces extreme, unstable weights — the
"error maximisation" problem described by Michaud (1989). This is the motivation for
Black-Litterman: it replaces the sample mean with a stable, economically anchored prior.

---

## 3. CAPM and market equilibrium

The Capital Asset Pricing Model (Sharpe, 1964; Lintner, 1965) implies that, in equilibrium,
the market-capitalisation-weighted portfolio is mean-variance efficient. Black and Litterman
use this in reverse: *assume* the market portfolio is optimal for a representative investor
with risk aversion `δ` and solve for the expected returns that make it so. The first-order
condition of `max w'μ − (δ/2) w'Σw` at `w = w_mkt` gives

```
Π = δ Σ w_mkt
```

`Π` is a vector of **implied excess returns** — a model quantity, not a forecast from
historical averages.

### Market weights

`w_mkt` are the current market capitalisations of the selected assets, normalised to sum
to one. Two consequences must be kept in mind:

1. The "market" is only the chosen universe (e.g. five technology stocks), not the global
   market portfolio assumed by CAPM. This is a strong assumption and the CLI prints it.
2. ETFs and indices have no market capitalisation; the user must supply weights with
   `--market-weights`.

### Risk aversion δ

* **Default `δ = 2.5`**, the value used by He & Litterman (1999). It is an assumption and
  is displayed on every screen.
* **Market-implied** alternative `δ = (E[r_m] − r_f) / σ_m²` (available in the library as
  `implied_risk_aversion`). It requires an assumed market excess return, so it moves the
  assumption rather than removing it.

---

## 4. Covariance estimation

| Estimator | Formula |
|---|---|
| `sample` | `S = X_c'X_c / (T − 1)`, annualised by `k` |
| `ledoit-wolf` | `Σ = s·m·I + (1 − s)·S₁` (Ledoit & Wolf, 2004) |

For Ledoit-Wolf, with `S₁ = X_c'X_c / T`, `m = tr(S₁)/N` and the normalised Frobenius norm
`‖A‖² = tr(AA')/N`:

```
d² = ‖S₁ − mI‖²
b² = min(d², (1/T²) Σ_t ‖x_t x_t' − S₁‖²)
s  = b² / d²
```

Shrinkage is recommended when the number of observations is small relative to the number
of assets; with `T ≤ N` the sample covariance is singular.

### Numerical safety

* **Symmetry** is checked to relative tolerance `1e-10`.
* **Positive semidefiniteness** is checked via the smallest eigenvalue (relative
  tolerance `1e-12`). Behaviour is configurable with `psd_policy`:
  * `raise` (default): stop with an explanation.
  * `repair`: eigenvalue clipping `Σ* = V·max(Λ, ε)·V'`, the Frobenius-nearest PSD matrix
    (Higham, 1988) with a small positive floor so that `Σ*` is invertible. The size of the
    repair `‖Σ* − Σ‖_F` is reported.
* **Condition number** `λ_max/λ_min` is reported; above `1e8` a warning is shown.
* **Linear algebra** uses Cholesky factorisations and solves (`scipy.linalg.cho_solve`);
  a matrix with condition number above `1e14` is treated as singular and rejected.

---

## 5. Black-Litterman

### Original papers

Black & Litterman (1990) introduced the approach in a Goldman Sachs Fixed Income Research
note; it was published as Black & Litterman (1991) in the *Journal of Fixed Income* and
extended to global portfolios in Black & Litterman (1992, *Financial Analysts Journal*).
The papers' contribution is to start from the equilibrium returns implied by market
weights (rather than from historical means) and to blend in an investor's views in
proportion to the confidence in each, producing expected returns and portfolios that
deviate from the market only in the direction of the views. The mixing follows Theil's
mixed-estimation approach; Bayesian readings and the explicit formulas used by most
implementations were set out later (e.g. He & Litterman, 1999; Satchell & Scowcroft, 2000;
Meucci, 2010; Walters, 2014).

### Model

```
Prior:   μ ~ N(Π, τΣ)
Views:   Pμ = Q + ε,   ε ~ N(0, Ω)
```

* `P` (`K × N`) picks the assets each view is about.
* `Q` (`K`) holds the view returns, in **excess** of `r_f`.
* `Ω` (`K × K`) is the view-uncertainty covariance.
* `τ` scales the uncertainty of the prior mean relative to the return covariance.

### Posterior mean

The canonical, precision-weighted form:

```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]
```

and the equivalent form obtained from the Woodbury identity:

```
μ_BL = Π + τΣP' (PτΣP' + Ω)⁻¹ (Q − PΠ)
```

Equilibrium computes the second form (a `K × K` solve, defined even when `Ω` is singular,
i.e. a view held with 100% confidence) and, whenever `Ω` is invertible, also the first;
it raises an error if they disagree beyond tolerance. The observed difference is reported
by `--verbose` (typically ~1e-16).

### Three covariance objects

| Object | Formula | Meaning |
|---|---|---|
| `Σ` | estimated | covariance of returns (equilibrium / historical) |
| `M` | `[(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ = τΣ − τΣP'(PτΣP' + Ω)⁻¹PτΣ` | uncertainty of the **estimate** `μ_BL` |
| `Σ_post` | `Σ + M` | posterior **predictive** covariance of returns (He & Litterman, 1999) |

Portfolios are optimised on `(μ_BL, Σ_post)`. `--verbose` shows `√M_ii` (standard error of
the mean estimate) alongside `√Σ_ii` and `√Σ_post,ii`.

### Views

| View | P row | Q |
|---|---|---|
| Absolute: "NVDA returns 8%" | `e_NVDA` | `0.08 − r_f` |
| Relative: "NVDA beats MSFT by 5%" | `e_NVDA − e_MSFT` | `0.05` |

Absolute views are entered as total returns and converted to excess returns; relative views
need no conversion because `r_f` cancels.

**Views are relative to equilibrium, not to zero.** A view moves the posterior only through
`Q − PΠ`. If equilibrium already implies that NVDA beats MSFT by 14.5%, the view "NVDA beats
MSFT by 5%" is *bearish* on NVDA relative to MSFT and will reduce the NVDA weight. The CLI
shows `PΠ` and the tilt `Q − PΠ` for every view.

### Ω — view uncertainty

There is no canonical mapping from a verbal "confidence" to `Ω`. Equilibrium implements
three explicit choices; `Ω` is diagonal (independent view errors) unless a full matrix is
supplied through the library API.

1. **He-Litterman proportional** (default when no confidence is given):
   `ω_k = p_k (τΣ) p_k'`. View variance equals the prior variance of the viewed portfolio.
   Under this choice the posterior mean does not depend on `τ` (tested).
2. **Idzorek confidence** (when `--confidence C` is given, `0 < C ≤ 1`). Idzorek (2005)
   defines confidence as the fraction of the way the unconstrained portfolio
   `w = (δΣ)⁻¹μ` moves from `w_mkt` toward the portfolio implied by a 100%-certain view.
   For a single view,
   ```
   w(ω) − w_mkt = τ p'(q − pΠ) / (δ (pτΣp' + ω))
   ```
   so the tilt relative to the `ω = 0` tilt is exactly `pτΣp' / (pτΣp' + ω)`. Setting this
   equal to `C` gives the closed form
   ```
   ω_k = ((1 − C_k) / C_k) · p_k (τΣ) p_k'
   ```
   Because the tilt is a scalar multiple of the full-confidence tilt, this coincides with
   Idzorek's least-squares calibration. `C = 1 ⇒ ω = 0`; `C → 0 ⇒ ω → ∞` (the view is
   ignored, so a 0% view is rejected rather than represented). Note that `C = 0.5`
   reproduces the He-Litterman choice. The test suite verifies numerically that the tilt
   equals `C` times the full tilt.
3. **Explicit** `--omega ω` per view.

`equilibrium views omega` and `portfolio --verbose` print `P`, `Q`, `Ω`, the prior view
variance `pτΣp'` and the implied confidence `pτΣp' / (pτΣp' + ω)` of every view.

### τ

`τ` scales `Σ` into the uncertainty of the equilibrium mean. The literature disagrees on its
value: Black & Litterman (1992) argue it should be small because the uncertainty of a mean
is much smaller than that of returns; common choices range from 0.025–0.05 to `1/T`.
The default is `τ = 0.05`. Under He-Litterman `Ω`, `τ` affects `Σ_post` but not `μ_BL`.

### Limiting behaviour (all tested)

* No views: `μ_BL = Π` exactly, `M = τΣ`, `Σ_post = (1 + τ)Σ`, and the unconstrained
  optimum `(δΣ_post)⁻¹Π = w_mkt / (1 + τ)` (He & Litterman, 1999). The fully-invested
  max-Sharpe portfolio is exactly `w_mkt`.
* Confidence → 0: posterior → prior. Confidence = 1: `Pμ_BL = Q` exactly.
* A view equal to its equilibrium value (`Q = PΠ`) changes nothing.
* Contradictory 100%-confidence views are rejected (singular system).

### Hand-verified example

`tests/test_black_litterman.py` contains a three-asset example worked on paper
(vols 20/30/15%, correlations 0.2, `w_mkt = (0.5, 0.3, 0.2)`, `δ = 2.5`, `τ = 0.05`, one
absolute view `q = 5%` with `ω = 0.0005`) with the expected posterior written as exact
fractions, e.g. `μ_C = (4/13)·0.0255 + (9/13)·0.05`. The implementation matches to 1e-14.

---

## 6. Performance statistics (historical)

| Statistic | Definition |
|---|---|
| Sharpe (Sharpe, 1994) | `mean(R − r_f,p) / std(R − r_f,p) · √k` |
| Downside deviation | `√(mean(min(R − MAR, 0)²))`, full-sample denominator, `MAR = r_f,p` |
| Sortino (Sortino & Price, 1994) | `mean(R − MAR)·k / (DD·√k)` |
| CAGR | `W_T^(k/T) − 1` |
| Max drawdown | `min_t (W_t / max_{s≤t} W_s − 1)` with the initial capital counted as a peak |

The ex-ante Sharpe ratio of a model portfolio is `w'μₓ / √(w'Σ_post w)`.

---

## 7. Backtesting

A walk-forward simulation without look-ahead in the estimation of `Σ` and `μ`:

1. Rebalance dates are the last trading day of each month or quarter.
2. At date `t` a strategy sees only prices up to and including the close of `t`, and the
   trailing `lookback` returns at the estimation frequency ending at `t`.
3. `Σ` is estimated from that window; target weights are computed and traded at the close.
4. From `t+1` the portfolio earns daily returns and weights drift:
   `w_i ← w_i(1 + R_i)/(1 + R_p)`.
5. Transaction costs: `c · Σ_i |w_target,i − w_drift,i|` with `c` in bps of traded notional
   (the initial purchase is included). Reported turnover is one-way, `½Σ|Δw|`.

Tests rewrite future prices and verify that past weights do not change, check that the
strategy never receives data after `t`, and reproduce the equal-weight simulation with an
independent implementation.

### Methodologies (alternatives, not a ranking)

Equal weight · Market-cap weight · Minimum variance · Max Sharpe (historical mean) ·
Black-Litterman (max-Sharpe on the posterior).

### Known biases, disclosed on every run

* **Market caps.** No free source provides historical market capitalisations. Weights at
  date `t` use `cap_now · P_t / P_now` — today's share count times the historical price —
  which uses future information about share issuance and buybacks.
* **Views.** Today's views applied at historical dates were formed with hindsight; a BL
  backtest with views is an illustration, not an out-of-sample test.
* **Survivorship.** The universe is chosen today; failed companies are absent.
* **Fallback.** When no asset has a positive expected excess return at a rebalance, the
  max-Sharpe portfolio is undefined; minimum variance is used and the count is reported.

---

## 8. Stress testing

**Historical scenarios** (observed): buy-and-hold over the window, so the portfolio return
is exactly `Σ w_i R_i` of cumulative asset returns. Windows are S&P 500 peak-to-trough
dates: dot-com (2000-03-24 → 2002-10-09), GFC (2007-10-09 → 2009-03-09), 2011
(2011-04-29 → 2011-10-03), COVID (2020-02-19 → 2020-03-23), 2022 (2022-01-03 → 2022-10-12),
or a custom window. Assets without prices for the whole window are refused — no proxy.

**Hypothetical shocks** (model): for shocks `s_K` to assets `K`,

* direct impact `Σ_{i∈K} w_i s_i`;
* covariance-propagated impact, moving the other assets by the conditional mean of a
  zero-mean Gaussian, `E[r_U | r_K = s_K] = Σ_UK Σ_KK⁻¹ s_K`.

Crises show fatter tails and higher correlations than a covariance estimated in normal
times, so both figures are approximations.

---

## References

* Black, F. and Litterman, R. (1990). *Asset Allocation: Combining Investor Views with
  Market Equilibrium.* Goldman, Sachs & Co., Fixed Income Research.
* Black, F. and Litterman, R. (1991). Asset Allocation: Combining Investor Views with Market
  Equilibrium. *Journal of Fixed Income*, 1(2), 7–18.
* Black, F. and Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts
  Journal*, 48(5), 28–43.
* Cornuéjols, G. and Tütüncü, R. (2007). *Optimization Methods in Finance.* Cambridge
  University Press.
* He, G. and Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios.*
  Goldman Sachs Investment Management Research.
* Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix.
  *Linear Algebra and its Applications*, 103, 103–118.
* Idzorek, T. M. (2005). *A Step-by-Step Guide to the Black-Litterman Model: Incorporating
  User-Specified Confidence Levels.* Working paper.
* Ledoit, O. and Wolf, M. (2004). A well-conditioned estimator for large-dimensional
  covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
* Lintner, J. (1965). The Valuation of Risk Assets and the Selection of Risky Investments in
  Stock Portfolios and Capital Budgets. *Review of Economics and Statistics*, 47(1), 13–37.
* Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
* Meucci, A. (2010). The Black-Litterman Approach: Original Model and Extensions. In
  *Encyclopedia of Quantitative Finance*, Wiley.
* Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized' Optimal?
  *Financial Analysts Journal*, 45(1), 31–42.
* Satchell, S. and Scowcroft, A. (2000). A demystification of the Black-Litterman model.
  *Journal of Asset Management*, 1(2), 138–150.
* Sharpe, W. F. (1964). Capital Asset Prices: A Theory of Market Equilibrium under
  Conditions of Risk. *Journal of Finance*, 19(3), 425–442.
* Sharpe, W. F. (1994). The Sharpe Ratio. *Journal of Portfolio Management*, 21(1), 49–58.
* Sortino, F. A. and Price, L. N. (1994). Performance Measurement in a Downside Risk
  Framework. *Journal of Investing*, 3(3), 59–64.
* Walters, J. (2014). *The Black-Litterman Model in Detail.* SSRN working paper.
