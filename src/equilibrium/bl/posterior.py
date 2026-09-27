"""Black-Litterman posterior (Black & Litterman, 1992; He & Litterman, 1999).

Model
-----
Prior on the mean excess return:      mu ~ N(Pi, tau Sigma)
Views:                                P mu = Q + eps,   eps ~ N(0, Omega)

Posterior mean (two algebraically equivalent forms):

  canonical  mu_BL = [(tau Sigma)^-1 + P' Omega^-1 P]^-1 [(tau Sigma)^-1 Pi + P' Omega^-1 Q]
  update     mu_BL = Pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P Pi)

The update form follows from the Woodbury identity. It needs only a K x K solve and remains
defined when Omega is singular (a view held with 100% confidence), where the canonical form
breaks down. Equilibrium computes the update form, and whenever Omega is invertible also
the canonical form, and verifies that they agree.

Three different covariance objects (never conflate them)
-------------------------------------------------------
  Sigma          covariance of returns, estimated from history.
  M              posterior covariance of the *estimate of the mean*:
                 M = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
                   = tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^-1 P tau Sigma
  Sigma_post     posterior predictive covariance of *returns*: Sigma_post = Sigma + M
                 (He & Litterman, 1999). With no views, M = tau Sigma and
                 Sigma_post = (1 + tau) Sigma.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from equilibrium.core.errors import NumericalError, ValidationError
from equilibrium.core.types import (
    CovarianceMatrix,
    ExpectedReturns,
    FloatArray,
    ReturnKind,
    Source,
)
from equilibrium.risk.numerics import condition_number, solve_spd

EQUIVALENCE_TOL = 1e-8


@dataclass(frozen=True)
class PosteriorResult:
    tickers: tuple[str, ...]
    prior: ExpectedReturns
    posterior: ExpectedReturns
    mean_uncertainty: FloatArray  # M
    predictive_cov: CovarianceMatrix  # Sigma + M
    P: FloatArray
    Q: FloatArray
    omega: FloatArray
    tau: float
    diagnostics: dict[str, float | str | bool] = field(default_factory=dict)

    @property
    def n_views(self) -> int:
        return int(self.P.shape[0])


def _validate_inputs(pi: FloatArray, cov: FloatArray, P: FloatArray, Q: FloatArray,
                     omega: FloatArray, tau: float) -> None:
    n = pi.shape[0]
    k = P.shape[0]
    if cov.shape != (n, n):
        raise ValidationError(f"Sigma is {cov.shape}, expected {(n, n)}.")
    if P.ndim != 2 or P.shape[1] != n:
        raise ValidationError(f"P is {P.shape}, expected (K, {n}).")
    if Q.shape != (k,):
        raise ValidationError(f"Q has shape {Q.shape}, expected ({k},).")
    if omega.shape != (k, k):
        raise ValidationError(f"Omega is {omega.shape}, expected {(k, k)}.")
    if not tau > 0:
        raise ValidationError("tau must be positive.")


def posterior_mean_update(pi: FloatArray, cov: FloatArray, P: FloatArray, Q: FloatArray,
                          omega: FloatArray, tau: float) -> FloatArray:
    """``Pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P Pi)`` via a Cholesky solve."""
    _validate_inputs(pi, cov, P, Q, omega, tau)
    if P.shape[0] == 0:
        return pi.copy()
    ts = tau * cov
    a = P @ ts @ P.T + omega
    try:
        x = solve_spd(0.5 * (a + a.T), Q - P @ pi, "P tau Sigma P' + Omega")
    except NumericalError as exc:
        raise NumericalError(
            f"{exc} With 100%-confidence views this happens when views are redundant or "
            "contradictory; lower a confidence or remove a duplicate view."
        ) from exc
    return pi + ts @ P.T @ x


def posterior_mean_canonical(pi: FloatArray, cov: FloatArray, P: FloatArray, Q: FloatArray,
                             omega: FloatArray, tau: float) -> FloatArray:
    """The textbook precision-weighted form. Requires invertible Omega and Sigma.

    ``(tau Sigma)^-1`` and ``Omega^-1`` are applied through Cholesky solves; no explicit
    inverse is formed except the precision matrix assembled column-by-column by ``cho_solve``.
    """
    _validate_inputs(pi, cov, P, Q, omega, tau)
    n = pi.shape[0]
    ts = tau * cov
    prior_precision = solve_spd(ts, np.eye(n), "tau Sigma")
    rhs = solve_spd(ts, pi, "tau Sigma")
    if P.shape[0]:
        omega_inv_p = solve_spd(omega, P, "Omega")
        omega_inv_q = solve_spd(omega, Q, "Omega")
        precision = prior_precision + P.T @ omega_inv_p
        rhs = rhs + P.T @ omega_inv_q
    else:
        precision = prior_precision
    return solve_spd(0.5 * (precision + precision.T), rhs, "posterior precision")


def posterior_mean_uncertainty(cov: FloatArray, P: FloatArray, omega: FloatArray, tau: float
                               ) -> FloatArray:
    """``M = tau Sigma - tau Sigma P' (P tau Sigma P' + Omega)^-1 P tau Sigma``."""
    ts = tau * cov
    if P.shape[0] == 0:
        return ts.copy()
    a = P @ ts @ P.T + omega
    m = ts - ts @ P.T @ solve_spd(0.5 * (a + a.T), P @ ts, "P tau Sigma P' + Omega")
    return 0.5 * (m + m.T)


def black_litterman(
    prior: ExpectedReturns,
    cov: CovarianceMatrix,
    P: FloatArray,
    Q: FloatArray,
    omega: FloatArray,
    tau: float,
) -> PosteriorResult:
    """Full Black-Litterman posterior with cross-checked formulations and diagnostics."""
    if prior.kind is not ReturnKind.EXCESS:
        raise ValidationError("The Black-Litterman prior must be expressed as excess returns.")
    pi = prior.aligned_to(cov.tickers)
    sigma = cov.values
    P = np.atleast_2d(np.asarray(P, dtype=np.float64)) if np.size(P) else np.zeros((0, pi.size))
    Q = np.asarray(Q, dtype=np.float64).reshape(-1)
    omega = np.atleast_2d(np.asarray(omega, dtype=np.float64)) if Q.size else np.zeros((0, 0))

    mu = posterior_mean_update(pi, sigma, P, Q, omega, tau)
    m = posterior_mean_uncertainty(sigma, P, omega, tau)

    diagnostics: dict[str, float | str | bool] = {
        "cond_sigma": condition_number(sigma),
        "formulation": "update (Woodbury)",
    }
    omega_invertible = Q.size == 0 or (
        np.all(np.linalg.eigvalsh(omega) > 1e-14) and condition_number(omega) < 1e12
    )
    if omega_invertible:
        mu_canon = posterior_mean_canonical(pi, sigma, P, Q, omega, tau)
        gap = float(np.max(np.abs(mu_canon - mu)))
        diagnostics["canonical_vs_update_max_abs_diff"] = gap
        scale = max(1.0, float(np.max(np.abs(mu))))
        if gap > EQUIVALENCE_TOL * scale * max(1.0, condition_number(sigma) * 1e-6):
            raise NumericalError(
                f"Canonical and update Black-Litterman formulas disagree by {gap:.2e}; the "
                "inputs are too ill-conditioned for a reliable posterior."
            )
    else:
        diagnostics["canonical_vs_update_max_abs_diff"] = "n/a (Omega singular: 100% view)"
    if Q.size:
        diagnostics["cond_view_system"] = condition_number(P @ (tau * sigma) @ P.T + omega)

    posterior = ExpectedReturns(
        cov.tickers, mu, kind=ReturnKind.EXCESS, source=Source.MODEL, label="posterior (mu_BL)"
    )
    predictive = CovarianceMatrix(
        cov.tickers, sigma + m, source=Source.MODEL, estimator=f"BL predictive ({cov.estimator})"
    )
    return PosteriorResult(
        tickers=cov.tickers, prior=prior, posterior=posterior, mean_uncertainty=m,
        predictive_cov=predictive, P=P, Q=Q, omega=omega, tau=tau, diagnostics=diagnostics,
    )
