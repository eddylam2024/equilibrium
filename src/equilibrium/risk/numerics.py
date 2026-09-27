"""Numerical safety for covariance matrices and linear algebra.

Positive-semidefinite (PSD) repair method
-----------------------------------------
When ``policy="repair"`` a non-PSD symmetric matrix is replaced by the matrix obtained by
clipping its eigenvalues from below:

    Sigma = V diag(lambda) V'   ->   Sigma* = V diag(max(lambda, eps)) V'

with ``eps = max(floor, 1e-10 * max(lambda))``. This is the nearest symmetric PSD matrix in
the Frobenius norm when ``eps = 0`` (Higham, 1988, "Computing a nearest symmetric positive
semidefinite matrix"); a small positive floor is used so that the result is also invertible.
The repair and its magnitude are always reported in the diagnostics, never applied silently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import linalg

from equilibrium.core.errors import NumericalError, ValidationError
from equilibrium.core.types import FloatArray

log = logging.getLogger(__name__)

SYMMETRY_TOL = 1e-10
PSD_RELATIVE_TOL = 1e-12
CONDITION_WARNING = 1e8
CONDITION_SINGULAR = 1e14


class PSDPolicy(StrEnum):
    REPAIR = "repair"
    RAISE = "raise"


@dataclass(frozen=True)
class MatrixDiagnostics:
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_psd: bool
    repaired: bool
    repair_frobenius_change: float

    @property
    def ill_conditioned(self) -> bool:
        return self.condition_number > CONDITION_WARNING

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "is_psd": self.is_psd,
            "repaired": self.repaired,
            "repair_frobenius_change": self.repair_frobenius_change,
            "ill_conditioned": self.ill_conditioned,
        }


def require_square(m: FloatArray, name: str = "matrix") -> None:
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValidationError(f"{name} must be square; got shape {m.shape}.")


def require_symmetric(m: FloatArray, name: str = "matrix", tol: float = SYMMETRY_TOL) -> None:
    require_square(m, name)
    scale = max(1.0, float(np.max(np.abs(m))))
    if not np.allclose(m, m.T, atol=tol * scale, rtol=0):
        raise NumericalError(f"{name} is not symmetric.")


def condition_number(m: FloatArray) -> float:
    """2-norm condition number (ratio of extreme singular values); ``inf`` if singular."""
    s = linalg.svdvals(m)
    if s[-1] <= 0:
        return float("inf")
    return float(s[0] / s[-1])


def is_psd(m: FloatArray, rel_tol: float = PSD_RELATIVE_TOL) -> bool:
    eig = linalg.eigvalsh(m)
    return bool(eig[0] >= -rel_tol * max(1.0, abs(eig[-1])))


def nearest_psd(m: FloatArray, floor: float = 0.0) -> FloatArray:
    """Eigenvalue-clipping projection onto the PSD cone (see module docstring)."""
    sym = 0.5 * (m + m.T)
    eigval, eigvec = linalg.eigh(sym)
    eps = max(floor, 1e-10 * max(float(eigval[-1]), 0.0))
    clipped = np.maximum(eigval, eps)
    out = (eigvec * clipped) @ eigvec.T
    return 0.5 * (out + out.T)


def validate_covariance(
    cov: FloatArray, policy: PSDPolicy = PSDPolicy.RAISE, name: str = "Covariance"
) -> tuple[FloatArray, MatrixDiagnostics]:
    """Check symmetry, PSD-ness and conditioning; repair or raise according to ``policy``."""
    cov = np.asarray(cov, dtype=np.float64)
    require_symmetric(cov, name)
    if np.any(np.diag(cov) < 0):
        raise NumericalError(f"{name} has negative variances on the diagonal.")
    eig = linalg.eigvalsh(cov)
    psd = bool(eig[0] >= -PSD_RELATIVE_TOL * max(1.0, abs(eig[-1])))
    repaired = False
    change = 0.0
    if not psd:
        if policy is PSDPolicy.RAISE:
            raise NumericalError(
                f"{name} is not positive semidefinite (min eigenvalue {eig[0]:.3e}). "
                "Set psd_policy: repair to project it onto the PSD cone, or use a shrinkage "
                "estimator (--estimator ledoit-wolf)."
            )
        fixed = nearest_psd(cov)
        change = float(np.linalg.norm(fixed - cov, "fro"))
        log.warning("%s was not PSD; repaired by eigenvalue clipping (||dSigma||_F=%.3e)",
                    name, change)
        cov, repaired = fixed, True
        eig = linalg.eigvalsh(cov)
    cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    diag = MatrixDiagnostics(
        min_eigenvalue=float(eig[0]),
        max_eigenvalue=float(eig[-1]),
        condition_number=cond,
        is_psd=psd,
        repaired=repaired,
        repair_frobenius_change=change,
    )
    return cov, diag


def cholesky_factor(m: FloatArray, name: str = "matrix") -> tuple[FloatArray, bool]:
    """Cholesky factor for ``scipy.linalg.cho_solve``; raises a clear error if singular."""
    try:
        return linalg.cho_factor(m, lower=True, check_finite=True)
    except linalg.LinAlgError as exc:
        raise NumericalError(
            f"{name} is singular or not positive definite; it cannot be factorised. "
            "This usually means perfectly collinear assets or too few observations."
        ) from exc


def solve_spd(a: FloatArray, b: FloatArray, name: str = "matrix") -> FloatArray:
    """Solve ``a x = b`` for symmetric positive-definite ``a`` via Cholesky (no inverse)."""
    if condition_number(a) > CONDITION_SINGULAR:
        raise NumericalError(
            f"{name} is numerically singular (condition number > {CONDITION_SINGULAR:.0e})."
        )
    return linalg.cho_solve(cholesky_factor(a, name), b)
