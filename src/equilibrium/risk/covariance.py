"""Covariance and correlation estimation.

Estimators
----------
``sample``       Unbiased sample covariance (denominator T-1), annualised by ``k``.
``ledoit-wolf``  Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
                 covariance matrices", J. Multivariate Analysis 88(2). Shrinks the sample
                 covariance toward a scaled identity ``m I`` with the optimal intensity
                 estimated from the data. Recommended when observations are few relative to
                 the number of assets.

All returned covariance matrices are *annualised* and are covariance, never correlation.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd

from equilibrium.core.errors import InsufficientHistoryError, ValidationError
from equilibrium.core.types import CovarianceMatrix, FloatArray, Source
from equilibrium.risk.numerics import PSDPolicy, validate_covariance


class CovarianceEstimator(StrEnum):
    SAMPLE = "sample"
    LEDOIT_WOLF = "ledoit-wolf"


def _clean_matrix(returns: pd.DataFrame) -> FloatArray:
    if returns.isna().any().any():
        raise ValidationError(
            "Returns contain NaN values; align and clean data before estimating covariance."
        )
    x = returns.to_numpy(dtype=np.float64)
    t, n = x.shape
    if t < 2:
        raise InsufficientHistoryError(f"Need at least 2 return observations; got {t}.")
    if t <= n:
        # Sample covariance of N assets from T <= N observations has rank < N.
        pass
    return x


def sample_covariance(returns: pd.DataFrame) -> FloatArray:
    """Unbiased periodic sample covariance ``S = X_c' X_c / (T - 1)``."""
    x = _clean_matrix(returns)
    xc = x - x.mean(axis=0)
    return (xc.T @ xc) / (x.shape[0] - 1)


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[FloatArray, float]:
    """Ledoit-Wolf (2004) shrinkage toward ``m I``; returns (periodic covariance, intensity).

    Following the paper's notation with the normalised Frobenius norm ``||A||^2 = tr(AA')/N``:
        S   = X_c' X_c / T
        m   = tr(S) / N
        d^2 = ||S - m I||^2
        b^2 = min(d^2, (1/T^2) sum_t ||x_t x_t' - S||^2)
        shrinkage = b^2 / d^2,   Sigma = shrinkage * m I + (1 - shrinkage) * S
    """
    x = _clean_matrix(returns)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = (xc.T @ xc) / t
    m = np.trace(s) / n
    d2 = np.sum((s - m * np.eye(n)) ** 2) / n
    if d2 <= 0:
        return s, 0.0
    b_bar2 = 0.0
    for row in xc:
        b_bar2 += np.sum((np.outer(row, row) - s) ** 2) / n
    b_bar2 /= t**2
    b2 = min(b_bar2, d2)
    shrink = float(b2 / d2)
    return shrink * m * np.eye(n) + (1.0 - shrink) * s, shrink


def estimate_covariance(
    returns: pd.DataFrame,
    periods_per_year: int,
    estimator: CovarianceEstimator = CovarianceEstimator.SAMPLE,
    psd_policy: PSDPolicy = PSDPolicy.RAISE,
) -> CovarianceMatrix:
    """Estimate an annualised covariance matrix with PSD validation and diagnostics."""
    t, n = returns.shape
    shrink: float | None = None
    if estimator is CovarianceEstimator.SAMPLE:
        periodic = sample_covariance(returns)
    else:
        periodic, shrink = ledoit_wolf_covariance(returns)
    annual = periodic * periods_per_year
    annual, diag = validate_covariance(annual, psd_policy)
    diagnostics: dict[str, float | str | bool] = dict(diag.as_dict())
    diagnostics["observations"] = float(t)
    diagnostics["assets"] = float(n)
    diagnostics["periods_per_year"] = float(periods_per_year)
    if shrink is not None:
        diagnostics["shrinkage_intensity"] = shrink
    if t <= n:
        diagnostics["warning"] = (
            f"Only {t} observations for {n} assets: the sample covariance is singular."
        )
    return CovarianceMatrix(
        tuple(str(c) for c in returns.columns),
        annual,
        source=Source.HISTORICAL,
        estimator=estimator.value,
        diagnostics=diagnostics,
    )


def correlation_from_covariance(cov: FloatArray) -> FloatArray:
    """``C = D^{-1/2} Sigma D^{-1/2}`` with ``D = diag(Sigma)``. Diagonal is exactly one."""
    sd = np.sqrt(np.diag(cov))
    if np.any(sd <= 0):
        raise ValidationError("Correlation undefined for an asset with zero variance.")
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)
