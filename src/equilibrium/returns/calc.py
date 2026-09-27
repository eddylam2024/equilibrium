"""Periodic return calculations and annualisation.

Conventions
-----------
* Simple (arithmetic) return:  R_t = P_t / P_{t-1} - 1
* Log (continuously compounded) return:  r_t = ln(P_t / P_{t-1})
* Simple returns are the default for portfolio work because they aggregate linearly
  across assets: the portfolio simple return is exactly ``w' R_t``. Log returns do not
  (``w' r_t`` is only an approximation), although they aggregate additively through time.
* Lower frequencies are obtained by resampling *prices* (last observation in each
  period) and then differencing, never by summing daily simple returns.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import Frequency


class ReturnMethod(StrEnum):
    SIMPLE = "simple"
    LOG = "log"


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices.empty:
        raise InsufficientHistoryError("Price table is empty.")
    if (prices <= 0).any().any():
        bad = prices.columns[(prices <= 0).any()].tolist()
        raise ValidationError(f"Non-positive prices found for: {', '.join(map(str, bad))}")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Arithmetic returns ``P_t / P_{t-1} - 1``; the first (undefined) row is dropped."""
    _validate_prices(prices)
    return (prices / prices.shift(1) - 1.0).iloc[1:]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded returns ``ln(P_t / P_{t-1})``; first row dropped."""
    _validate_prices(prices)
    ratio = (prices / prices.shift(1)).iloc[1:]
    return pd.DataFrame(np.log(ratio.to_numpy()), index=ratio.index, columns=ratio.columns)


def resample_prices(prices: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    """Sample prices at the end of each period (last available observation).

    Each row keeps the *actual* date of its observation (e.g. Fri 29 Jan, not the calendar
    month-end label 31 Jan), so timestamps never point past the information they contain.
    """
    rule = frequency.pandas_rule
    if rule is None:
        return prices
    period = "M" if frequency is Frequency.MONTHLY else rule
    prices = prices.dropna(how="all")
    return prices.groupby(pd.DatetimeIndex(prices.index).to_period(period)).tail(1)


def periodic_returns(
    prices: pd.DataFrame,
    frequency: Frequency = Frequency.MONTHLY,
    method: ReturnMethod = ReturnMethod.SIMPLE,
) -> pd.DataFrame:
    """Resample prices to ``frequency`` and compute returns with ``method``."""
    sampled = resample_prices(prices, frequency)
    if len(sampled) < 2:
        raise InsufficientHistoryError(
            f"Need at least two {frequency.value} price observations; got {len(sampled)}."
        )
    fn = simple_returns if method is ReturnMethod.SIMPLE else log_returns
    return fn(sampled)


def annualize_mean(periodic_mean: pd.Series | np.ndarray | float, periods_per_year: int):
    """Arithmetic annualisation of a mean periodic return: ``mu_annual = k * mu_periodic``.

    This is the scaling consistent with annualising covariance by ``k`` (it is exact for
    i.i.d. log returns and the standard approximation for simple returns).
    """
    return periodic_mean * periods_per_year


def annualize_volatility(periodic_vol: pd.Series | np.ndarray | float, periods_per_year: int):
    """Square-root-of-time scaling: ``sigma_annual = sqrt(k) * sigma_periodic``."""
    return periodic_vol * np.sqrt(periods_per_year)


def periodic_rate(annual_rate: float, periods_per_year: int) -> float:
    """Convert an annual effective rate to the equivalent compounding periodic rate."""
    if annual_rate <= -1:
        raise ValidationError("Annual rate must exceed -100%.")
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def cumulative_wealth(returns: pd.Series | pd.DataFrame, initial: float = 1.0):
    """Wealth index ``W_t = W_0 * prod(1 + R_s)`` from simple returns."""
    return initial * (1.0 + returns).cumprod()
