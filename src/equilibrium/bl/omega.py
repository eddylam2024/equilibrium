"""View-uncertainty matrix Omega.

There is no canonical mapping from a verbal "confidence" to Omega. Equilibrium supports three
explicit, documented choices, and Omega is always diagonal unless the user supplies a full
matrix (a diagonal Omega means view errors are assumed mutually independent).

1. He-Litterman proportional (default when no confidence is given)
       omega_k = p_k (tau Sigma) p_k'
   View uncertainty equals the prior uncertainty of the viewed portfolio (He & Litterman,
   1999; also Meucci, 2010). Under this choice the posterior does not depend on tau.

2. Idzorek confidence (used when a confidence C_k in (0, 1] is given)
   Idzorek (2005) defines C_k as the fraction of the way the *unconstrained* optimal
   portfolio ``w = (delta Sigma)^{-1} mu`` moves from ``w_mkt`` toward the portfolio it
   would hold with 100% certainty in view k alone. For a single view with variance omega:
       mu(omega) - Pi = tau Sigma p' (q - p Pi) / (p tau Sigma p' + omega)
       w(omega) - w_mkt = tau p' (q - p Pi) / (delta (p tau Sigma p' + omega))
   so the tilt relative to the omega = 0 tilt is exactly  p tau Sigma p' / (p tau Sigma p' + omega).
   Setting that ratio equal to C_k and solving gives the closed form used here:
       omega_k = ((1 - C_k) / C_k) * p_k (tau Sigma) p_k'
   It reproduces Idzorek's numerical least-squares calibration exactly (the tilt is a
   scalar multiple of the 100% tilt, so the least-squares fit is exact). C = 1 gives
   omega = 0 (view held with certainty); C -> 0 gives omega -> infinity (view ignored),
   which is why a 0% confidence is rejected rather than represented. Note that C = 0.5
   reproduces the He-Litterman proportional choice.

3. Explicit omega per view, or a full user-supplied K x K matrix (validated PSD).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from equilibrium.bl.views import View
from equilibrium.core.errors import ValidationError
from equilibrium.core.types import FloatArray
from equilibrium.risk.numerics import require_symmetric


class OmegaMethod(StrEnum):
    HE_LITTERMAN = "he-litterman"
    IDZOREK = "idzorek"
    EXPLICIT = "explicit"


def prior_view_variance(P: FloatArray, cov: FloatArray, tau: float) -> FloatArray:
    """Vector of ``p_k (tau Sigma) p_k'`` for each view row."""
    return np.einsum("ki,ij,kj->k", P, tau * cov, P)


def idzorek_omega(confidence: float, prior_variance: float) -> float:
    if not 0.0 < confidence <= 1.0:
        raise ValidationError(
            f"Confidence must be in (0, 1]; got {confidence}. A 0% confidence view has no "
            "effect on the posterior; remove it instead."
        )
    return (1.0 - confidence) / confidence * prior_variance


def build_omega(
    views: Sequence[View], P: FloatArray, cov: FloatArray, tau: float
) -> tuple[FloatArray, list[OmegaMethod]]:
    """Diagonal Omega from per-view settings; returns (Omega, method used for each view)."""
    prior = prior_view_variance(P, cov, tau)
    diag = np.zeros(len(views))
    methods: list[OmegaMethod] = []
    for k, v in enumerate(views):
        if v.omega is not None:
            diag[k] = v.omega
            methods.append(OmegaMethod.EXPLICIT)
        elif v.confidence is not None:
            diag[k] = idzorek_omega(v.confidence, float(prior[k]))
            methods.append(OmegaMethod.IDZOREK)
        else:
            diag[k] = prior[k]
            methods.append(OmegaMethod.HE_LITTERMAN)
    return np.diag(diag), methods


def validate_user_omega(omega: FloatArray, n_views: int) -> FloatArray:
    """Validate a full user-supplied Omega: K x K, symmetric, positive semidefinite."""
    omega = np.asarray(omega, dtype=np.float64)
    if omega.shape != (n_views, n_views):
        raise ValidationError(f"Omega must be {n_views}x{n_views}; got {omega.shape}.")
    require_symmetric(omega, "Omega")
    if np.linalg.eigvalsh(omega)[0] < -1e-12:
        raise ValidationError("Omega must be positive semidefinite.")
    return omega


def implied_confidence(omega_k: float, prior_variance: float) -> float:
    """Inverse of the Idzorek map: the confidence equivalent to a given omega."""
    return prior_variance / (prior_variance + omega_k)
