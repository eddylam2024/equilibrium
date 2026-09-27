"""Shared deterministic fixtures. No test in this suite touches the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equilibrium.core.types import CovarianceMatrix, Weights

TICKERS3 = ("AAA", "BBB", "CCC")


@pytest.fixture
def cov3() -> CovarianceMatrix:
    """sigma = (20%, 30%, 15%), all pairwise correlations 0.2."""
    s = np.array([
        [0.0400, 0.0120, 0.0060],
        [0.0120, 0.0900, 0.0090],
        [0.0060, 0.0090, 0.0225],
    ])
    return CovarianceMatrix(TICKERS3, s)


@pytest.fixture
def w_mkt3() -> Weights:
    return Weights(TICKERS3, np.array([0.5, 0.3, 0.2]))


def make_prices(n_days: int = 800, seed: int = 7, tickers=("AAA", "BBB", "CCC", "DDD")
                ) -> pd.DataFrame:
    """Deterministic correlated geometric random walk on business days."""
    rng = np.random.default_rng(seed)
    k = len(tickers)
    corr = 0.3 * np.ones((k, k)) + 0.7 * np.eye(k)
    vols = np.linspace(0.15, 0.40, k) / np.sqrt(252)
    cov = corr * np.outer(vols, vols)
    drift = np.linspace(0.04, 0.12, k) / 252
    shocks = rng.multivariate_normal(drift, cov, size=n_days)
    idx = pd.bdate_range("2015-01-01", periods=n_days + 1)
    prices = 100 * np.exp(np.vstack([np.zeros(k), np.cumsum(shocks, axis=0)]))
    return pd.DataFrame(prices, index=idx, columns=list(tickers))


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()
