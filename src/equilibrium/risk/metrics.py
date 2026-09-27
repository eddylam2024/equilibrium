"""Ex-ante portfolio risk and ex-post performance statistics.

Ex-ante (model) quantities use an annualised covariance matrix Sigma:
    portfolio variance            sigma_p^2 = w' Sigma w
    portfolio volatility          sigma_p   = sqrt(w' Sigma w)
    marginal contribution (MCR)   d sigma_p / d w = Sigma w / sigma_p
    component contribution (CCR)  CCR_i = w_i * MCR_i,     sum_i CCR_i = sigma_p (Euler)

Ex-post (historical) statistics use a periodic simple-return series with ``k`` periods
per year and an annual risk-free rate ``r_f`` converted to the periodic rate
``r_f,p = (1 + r_f)^(1/k) - 1``:
    Sharpe  = mean(R - r_f,p) / std(R - r_f,p) * sqrt(k)                 (Sharpe, 1994)
    Downside deviation (MAR = r_f,p) = sqrt( mean( min(R - MAR, 0)^2 ) ) (full-sample N)
    Sortino = mean(R - MAR) * k / (DD * sqrt(k))                        (Sortino & Price, 1994)
    CAGR    = W_T^(k / T) - 1
    Max drawdown = min_t (W_t / max_{s<=t} W_s - 1)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import FloatArray
from equilibrium.returns.calc import cumulative_wealth, periodic_rate


def _check_dims(w: FloatArray, cov: FloatArray) -> tuple[FloatArray, FloatArray]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape != (w.size, w.size):
        raise ValidationError(
            f"Dimension mismatch: {w.size} weights vs covariance of shape {cov.shape}."
        )
    return w, cov


def portfolio_variance(w: FloatArray, cov: FloatArray) -> float:
    w, cov = _check_dims(w, cov)
    return float(w @ cov @ w)


def portfolio_volatility(w: FloatArray, cov: FloatArray) -> float:
    var = portfolio_variance(w, cov)
    if var < -1e-14:
        raise ValidationError(f"Negative portfolio variance {var:.3e}: covariance is not PSD.")
    return float(np.sqrt(max(var, 0.0)))


def marginal_risk_contribution(w: FloatArray, cov: FloatArray) -> FloatArray:
    w, cov = _check_dims(w, cov)
    vol = portfolio_volatility(w, cov)
    if vol == 0:
        raise ValidationError("Marginal risk is undefined for a zero-volatility portfolio.")
    return (cov @ w) / vol


def component_risk_contribution(w: FloatArray, cov: FloatArray) -> FloatArray:
    w, cov = _check_dims(w, cov)
    return w * marginal_risk_contribution(w, cov)


def percent_risk_contribution(w: FloatArray, cov: FloatArray) -> FloatArray:
    return component_risk_contribution(w, cov) / portfolio_volatility(w, cov)


def ex_ante_sharpe(expected_excess_return: float, volatility: float) -> float:
    """Sharpe ratio of a *model* expected excess return; the numerator must be excess."""
    if volatility <= 0:
        raise ValidationError("Sharpe ratio undefined for zero volatility.")
    return expected_excess_return / volatility


# ---------------------------------------------------------------- ex-post statistics


def _series(returns: pd.Series | FloatArray) -> pd.Series:
    s = pd.Series(returns, dtype=np.float64).dropna()
    if len(s) < 2:
        raise InsufficientHistoryError("At least two return observations are required.")
    return s


def sharpe_ratio(returns: pd.Series | FloatArray, risk_free_rate: float, periods_per_year: int
                 ) -> float:
    r = _series(returns) - periodic_rate(risk_free_rate, periods_per_year)
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series | FloatArray, mar_periodic: float = 0.0) -> float:
    r = _series(returns)
    shortfall = np.minimum(r.to_numpy() - mar_periodic, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino_ratio(returns: pd.Series | FloatArray, risk_free_rate: float, periods_per_year: int
                  ) -> float:
    mar = periodic_rate(risk_free_rate, periods_per_year)
    r = _series(returns)
    dd = downside_deviation(r, mar)
    if dd == 0:
        return float("nan")
    return float((r.mean() - mar) * periods_per_year / (dd * np.sqrt(periods_per_year)))


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = cumulative_wealth(_series(returns))
    peak = np.maximum(wealth.cummax(), 1.0)  # the initial wealth of 1 is also a peak
    return wealth / peak - 1.0


def max_drawdown(returns: pd.Series | FloatArray) -> float:
    return float(drawdown_series(_series(returns)).min())


def cagr(returns: pd.Series | FloatArray, periods_per_year: int) -> float:
    r = _series(returns)
    total = float(np.prod(1.0 + r.to_numpy()))
    years = len(r) / periods_per_year
    if total <= 0:
        return -1.0
    return total ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series | FloatArray, periods_per_year: int) -> float:
    return float(_series(returns).std(ddof=1) * np.sqrt(periods_per_year))


@dataclass(frozen=True)
class PerformanceSummary:
    """Historical (ex-post) performance of a return series. All values are observations."""

    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    downside_deviation: float
    observations: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def performance_summary(
    returns: pd.Series, risk_free_rate: float, periods_per_year: int
) -> PerformanceSummary:
    r = _series(returns)
    mar = periodic_rate(risk_free_rate, periods_per_year)
    return PerformanceSummary(
        total_return=float(np.prod(1.0 + r.to_numpy()) - 1.0),
        cagr=cagr(r, periods_per_year),
        volatility=annualized_volatility(r, periods_per_year),
        sharpe=sharpe_ratio(r, risk_free_rate, periods_per_year),
        sortino=sortino_ratio(r, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(r),
        downside_deviation=downside_deviation(r, mar) * np.sqrt(periods_per_year),
        observations=len(r),
    )
