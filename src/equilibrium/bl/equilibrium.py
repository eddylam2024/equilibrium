"""Market equilibrium (reverse optimisation).

If the representative investor holds the market portfolio ``w_mkt`` and maximises
``w' mu - (delta / 2) w' Sigma w``, the first-order condition gives the implied equilibrium
*excess* returns (Black & Litterman, 1992; He & Litterman, 1999):

    Pi = delta * Sigma * w_mkt

``Sigma`` is the covariance matrix of excess returns (not the correlation matrix) and must
use the same annualisation as ``delta``. ``Pi`` is a model quantity, not a forecast derived
from historical means.

Risk aversion ``delta``
-----------------------
* Default: 2.5, the value used in He & Litterman (1999) for a global equity market.
* Market-implied alternative: ``delta = (E[r_m] - r_f) / sigma_m^2`` from an assumed market
  excess return and the variance of the market portfolio (Black & Litterman, 1992 discuss
  calibrating it from the market Sharpe ratio). Both choices are assumptions and the CLI
  reports which one is in force.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from equilibrium.core.errors import ValidationError
from equilibrium.core.types import (
    CovarianceMatrix,
    ExpectedReturns,
    ReturnKind,
    Source,
    Weights,
)

DEFAULT_RISK_AVERSION = 2.5
DEFAULT_TAU = 0.05


def market_cap_weights(market_caps: Mapping[str, float], tickers: Sequence[str]) -> Weights:
    """Normalise market capitalisations to weights aligned with ``tickers``."""
    missing = [t for t in tickers if t not in market_caps]
    if missing:
        raise ValidationError(f"Missing market capitalisation for: {', '.join(missing)}")
    caps = np.array([float(market_caps[t]) for t in tickers])
    if np.any(~np.isfinite(caps)) or np.any(caps <= 0):
        bad = [t for t, c in zip(tickers, caps, strict=True) if not np.isfinite(c) or c <= 0]
        raise ValidationError(f"Market capitalisation must be positive for: {', '.join(bad)}")
    return Weights.normalized(tickers, caps)


def implied_risk_aversion(market_excess_return: float, market_variance: float) -> float:
    """``delta = (E[r_m] - r_f) / sigma_m^2``."""
    if market_variance <= 0:
        raise ValidationError("Market variance must be positive.")
    if market_excess_return <= 0:
        raise ValidationError(
            "Implied risk aversion requires a positive market excess return."
        )
    return market_excess_return / market_variance


def equilibrium_returns(
    cov: CovarianceMatrix, market_weights: Weights, risk_aversion: float
) -> ExpectedReturns:
    """Implied equilibrium excess returns ``Pi = delta Sigma w_mkt``."""
    if risk_aversion <= 0:
        raise ValidationError("Risk aversion (delta) must be positive.")
    market_weights.require_fully_invested()
    w = market_weights.aligned_to(cov.tickers)
    pi = risk_aversion * cov.values @ w
    return ExpectedReturns(
        cov.tickers, pi, kind=ReturnKind.EXCESS, source=Source.MODEL, label="equilibrium (Pi)"
    )
