"""Recover the implicit market weights and variance from covariance and beta.

No convex solver or factor-portfolio weights are required. The factor path
supports positive diagonal specific variance without forming an NxN matrix.
Both paths return RAW implied weights, never clipped or renormalized.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class RecoveryNumericalError(RuntimeError):
    """The linear solve or covariance/beta reconstruction was not accurate enough."""


@dataclass(frozen=True)
class MarketRecovery:
    weights: FloatArray
    market_variance: float
    beta_recomputed: FloatArray
    full_investment_consistent: bool
    long_only_consistent: bool
    diagnostics: dict[str, float]
    method: str

    @property
    def market_volatility(self) -> float:
        return float(np.sqrt(self.market_variance))

    @property
    def market_consistent(self) -> bool:
        """Consistent with a fully invested long-only market, within tolerance.

        This is a compatibility check, not proof of the vendor's market identity.
        Raw weights may still contain tiny negative roundoff values.
        """
        return self.full_investment_consistent and self.long_only_consistent


def _beta(value: ArrayLike, n: int) -> FloatArray:
    beta = np.asarray(value, dtype=float)
    if beta.shape == (n, 1):
        beta = beta[:, 0]
    if beta.shape != (n,) or not np.isfinite(beta).all() or not np.any(beta != 0):
        raise ValueError("predicted_beta must be a finite nonzero N-vector or Nx1 column")
    return beta


def _symmetric(value: ArrayLike, name: str) -> tuple[FloatArray, float]:
    matrix = np.asarray(value, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()):
        raise ValueError(f"{name} must be a nonempty finite square matrix")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * matrix + 0.5 * matrix.T, scale


def _tolerance(value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError("consistency_tolerance must be finite and positive")


def _summarize(
    beta: FloatArray,
    v: FloatArray,
    multiply: Callable[[FloatArray], FloatArray],
    *,
    method: str,
    consistency_tolerance: float,
) -> MarketRecovery:
    denominator = float(beta @ v)
    if not np.isfinite(denominator) or denominator <= 0 or not np.isfinite(v).all():
        raise RecoveryNumericalError("beta.T @ solve(Sigma, beta) must be finite and positive")
    variance = 1.0 / denominator
    weights = v / denominator
    if not np.isfinite(variance) or not np.isfinite(weights).all():
        raise RecoveryNumericalError("nonfinite recovered variance or weights")
    sigma_w = multiply(weights)
    actual_variance = float(weights @ sigma_w)
    if not np.isfinite(actual_variance) or actual_variance <= 0:
        raise RecoveryNumericalError("recovered portfolio has invalid variance")
    beta_recomputed = sigma_w / actual_variance
    beta_scale = float(np.max(np.abs(beta)))
    solve_error = float(np.max(np.abs(multiply(v) - beta))) / beta_scale
    beta_error = float(np.max(np.abs(beta_recomputed - beta)))
    variance_error = abs(actual_variance / variance - 1.0)
    if (not np.isfinite(beta_recomputed).all()
            or not np.isfinite([solve_error, beta_error, variance_error]).all()
            or solve_error > 1e-9 or beta_error / beta_scale > 1e-9 or variance_error > 1e-9):
        raise RecoveryNumericalError(
            f"linear solve/reconstruction failed: solve={solve_error:.3g}, "
            f"beta={beta_error:.3g}, variance={variance_error:.3g}"
        )

    budget_error = abs(float(weights.sum()) - 1.0)
    negative_mass = float(-np.minimum(weights, 0).sum())
    diagnostics = {
        "weight_sum": float(weights.sum()),
        "full_investment_error": budget_error,
        "minimum_weight": float(weights.min()),
        "negative_weight_mass": negative_mass,
        "market_weighted_beta": float(weights @ beta),
        "linear_solve_relative_residual": solve_error,
        "beta_reconstruction_max_error": beta_error,
        "variance_relative_error": variance_error,
    }
    return MarketRecovery(
        weights=weights, market_variance=variance, beta_recomputed=beta_recomputed,
        full_investment_consistent=budget_error <= consistency_tolerance,
        long_only_consistent=negative_mass <= consistency_tolerance,
        diagnostics=diagnostics, method=method,
    )


def recover_from_covariance(
    covariance: ArrayLike,
    predicted_beta: ArrayLike,
    *,
    consistency_tolerance: float = 1e-8,
) -> MarketRecovery:
    """Solve Sigma v = beta, then s = 1/(beta.T v), w = s*v.

    Requires a positive-definite stock covariance and nonzero beta. Cholesky
    checks positive definiteness; singular matrices are not pseudoinverted.
    Incompatible budget/long-only weights are returned with false diagnostic
    flags rather than silently normalized or clipped. Units of s match Sigma.

    Dense reference path: O(N^3) arithmetic, O(N^2) memory.
    """
    _tolerance(consistency_tolerance)
    sigma, scale = _symmetric(covariance, "covariance")
    beta = _beta(predicted_beta, sigma.shape[0])
    try:
        chol = np.linalg.cholesky(sigma / scale)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite; no pseudoinverse is used") from exc
    # Factor and solve, never construct an inverse.
    v = np.linalg.solve(chol.T, np.linalg.solve(chol, beta)) / scale
    return _summarize(
        beta, v, lambda value: sigma @ value,
        method="dense_cholesky", consistency_tolerance=consistency_tolerance,
    )


def recover_from_factors(
    loadings: ArrayLike,
    factor_covariance: ArrayLike,
    specific_covariance: ArrayLike,
    predicted_beta: ArrayLike,
    *,
    consistency_tolerance: float = 1e-8,
) -> MarketRecovery:
    """Recover w,s from X,F,diagonal D,beta without materializing Sigma.

    F may be singular PSD; D may be a diagonal NxN matrix or its diagonal as
    an N-vector. Every specific variance must be strictly positive. Beta may
    be an N-vector or an Nx1 column; output weights are always an N-vector.
    For correlated D (or zero specific variances), form Sigma and use
    recover_from_covariance instead. No off-diagonal terms are discarded.

    A PSD square root F=C C.T gives U=X C. Woodbury then uses only the KxK
    system I + (D^-1/2 U).T (D^-1/2 U). No inverse of F is needed.
    Cost O(N*K^2 + K^3); additional memory O(N*K + K^2). An input NxN D
    also requires an O(N^2) diagonal-structure check; an N-vector avoids it.
    """
    _tolerance(consistency_tolerance)
    X = np.asarray(loadings, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("loadings must be a nonempty finite NxK matrix")
    n, k = X.shape
    F, factor_scale = _symmetric(factor_covariance, "factor_covariance")
    if F.shape != (k, k):
        raise ValueError("factor_covariance shape does not match loadings")
    specific = np.asarray(specific_covariance, dtype=float)
    d = np.diag(specific).copy() if specific.shape == (n, n) else specific
    if d.shape != (n,) or not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("specific_covariance must contain strictly positive specific variances")
    if specific.ndim == 2 and np.count_nonzero(specific) != n:
        raise ValueError("specific_covariance must be diagonal; use the dense path for correlated D")
    beta = _beta(predicted_beta, n)
    eigenvalues, eigenvectors = np.linalg.eigh(F / factor_scale)
    if eigenvalues.min() < -1e-12:
        raise ValueError("factor_covariance must be positive semidefinite")
    # Only numerical negative eigenvalues within PSD tolerance are rounded
    # to zero to construct the square root. No inverse or ridge is used.
    C = eigenvectors * (np.sqrt(np.maximum(eigenvalues, 0)) * np.sqrt(factor_scale))
    U = X @ C
    sqrt_d = np.sqrt(d)
    B = U / sqrt_d[:, None]
    small_system = np.eye(k) + B.T @ B
    try:
        chol = np.linalg.cholesky(small_system)
    except np.linalg.LinAlgError as exc:
        raise RecoveryNumericalError("factor-space linear system is numerically ill-conditioned") from exc

    def solve(rhs: FloatArray) -> FloatArray:
        whitened = rhs / sqrt_d
        coefficient = np.linalg.solve(chol.T, np.linalg.solve(chol, B.T @ whitened))
        return (whitened - B @ coefficient) / sqrt_d

    def multiply(value: FloatArray) -> FloatArray:
        # Use original F for residual checks, including its supplied precision.
        return X @ (F @ (X.T @ value)) + d * value

    v = solve(beta)
    # Woodbury can suffer cancellation for very small specific variance.
    # Refine against the original model; otherwise fail the numerical checks.
    for _ in range(3):
        residual = beta - multiply(v)
        if np.max(np.abs(residual)) <= 1e-13 * np.max(np.abs(beta)):
            break
        candidate = v + solve(residual)
        if np.max(np.abs(beta - multiply(candidate))) >= np.max(np.abs(residual)):
            break
        v = candidate
    return _summarize(
        beta, v, multiply, method="factor_woodbury",
        consistency_tolerance=consistency_tolerance,
    )
