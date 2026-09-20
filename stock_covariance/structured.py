"""Matrix-free stock beta adjustment and numeric risk calculations.

Pure NumPy; no solver objects or solver imports. Keep the dense beta_guard API as a
reference/backward-compatible path. No NxN matrix is built except in to_dense().
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .beta_guard import NumericalError, _covariance, _weights, calibrate_betas

FloatArray = NDArray[np.float64]


def _vector(value: ArrayLike, n: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape == (n, 1):
        array = array[:, 0]
    if array.shape != (n,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite N-vector or Nx1 column")
    return array


def _readonly(value: FloatArray) -> FloatArray:
    snapshot = np.array(value, dtype=float, copy=True)
    snapshot.flags.writeable = False
    return snapshot


@dataclass(frozen=True)
class StructuredCovariance:
    """Immutable snapshots describing Sigma_star = A Sigma A.T.

    factor_risk_loadings is U=X C with F=C C.T, NOT Barra's raw loading X.
    A=I+delta*w.T is never materialized. Input asset order is preserved.
    """

    factor_risk_loadings: FloatArray
    specific_variances: FloatArray
    market_weights: FloatArray
    beta_before: FloatArray
    beta_target: FloatArray
    beta_after: FloatArray
    delta: FloatArray
    market_variance: float
    updated_market_variance: float
    shift: float
    iterations: int
    diagnostics: dict[str, float]

    @property
    def n_assets(self) -> int:
        return self.market_weights.size

    @property
    def n_factors(self) -> int:
        return self.factor_risk_loadings.shape[1]

    @property
    def storage_bytes(self) -> int:
        """Stored NumPy array bytes, not process peak memory."""
        return sum(value.nbytes for value in (
            self.factor_risk_loadings, self.specific_variances, self.market_weights,
            self.beta_before, self.beta_target, self.beta_after, self.delta,
        ))

    def transform(self, exposure: ArrayLike) -> FloatArray:
        """A.T @ exposure, in O(N). exposure can be weights or dollar holdings."""
        p = _vector(exposure, self.n_assets, "exposure")
        return p + self.market_weights * float(self.delta @ p)

    def matvec(self, exposure: ArrayLike) -> FloatArray:
        """Sigma_star @ exposure, in O(N*K), without a dense covariance."""
        z = self.transform(exposure)
        U = self.factor_risk_loadings
        base = U @ (U.T @ z) + self.specific_variances * z
        return base + self.delta * float(self.market_weights @ base)

    def variance(self, exposure: ArrayLike) -> float:
        """exposure.T @ Sigma_star @ exposure as a sum of nonnegative terms."""
        z = self.transform(exposure)
        factors = self.factor_risk_loadings.T @ z
        return float(factors @ factors + self.specific_variances @ (z * z))

    def gradient(self, exposure: ArrayLike) -> FloatArray:
        """Gradient of variance with respect to exposure: 2*Sigma_star*exposure."""
        return 2 * self.matvec(exposure)

    def diagonal(self) -> FloatArray:
        """All marginal asset variances, O(N*K); no full matrix."""
        U = self.factor_risk_loadings
        base = np.einsum("ij,ij->i", U, U) + self.specific_variances
        return base + self.market_variance * (
            2 * self.beta_before * self.delta + self.delta * self.delta
        )

    def to_dense(self, indices: ArrayLike | None = None) -> FloatArray:
        """Materialize Sigma_star, or its selected principal submatrix.

        Full: O(N^2*K) work and O(N^2) output memory. Selected M assets:
        O(M^2*K) work and O(M^2) output. Calibration always stays on the FULL
        original universe; selection never renormalizes the market weights.
        """
        if indices is None:
            selected = np.arange(self.n_assets)
        else:
            selected = np.asarray(indices)
            if (selected.ndim != 1 or selected.size == 0
                    or not np.issubdtype(selected.dtype, np.integer)
                    or np.any(selected < 0) or np.any(selected >= self.n_assets)
                    or np.unique(selected).size != selected.size):
                raise ValueError("indices must be a nonempty 1-D array of unique valid integer positions")
        U = self.factor_risk_loadings[selected]
        beta, delta = self.beta_before[selected], self.delta[selected]
        result = U @ U.T
        result[np.diag_indices_from(result)] += self.specific_variances[selected]
        result += self.market_variance * (
            np.outer(beta, delta) + np.outer(delta, beta) + np.outer(delta, delta)
        )
        return 0.5 * result + 0.5 * result.T


def adjust_from_factors(
    loadings: ArrayLike,
    factor_covariance: ArrayLike,
    specific_covariance: ArrayLike,
    market_weights: ArrayLike,
    *,
    lower_bound: float = 0.15,
    tolerance: float = 1e-12,
    max_iterations: int = 128,
) -> StructuredCovariance:
    """Same stock-covariance beta rule as adjust_covariance, without NxN arrays.

    X is NxK, F is PSD KxK, D is a nonnegative variance vector or diagonal
    NxN matrix. Dense correlated D must use the legacy dense API. w accepts
    (N,) or (N,1), must be long-only and sum to one; no implicit normalization.

    Preparation O(N*K^2 + K^3 + N*iterations), storage O(N*K + K^2).
    Checking an input NxN D also costs O(N^2); pass its vector to avoid this.
    Tiny negative F eigenvalues within 1e-12 relative tolerance are rounded
    to zero for the PSD square root; material indefiniteness is rejected.
    """
    X = np.asarray(loadings, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("loadings must be a nonempty finite NxK matrix")
    n, k = X.shape
    F = _covariance(factor_covariance, False)
    if F.shape != (k, k):
        raise ValueError("factor_covariance shape must match loadings")
    scale = max(float(np.max(np.abs(F))), np.finfo(float).tiny)
    eig, basis = np.linalg.eigh(F / scale)
    if eig.min() < -1e-12:
        raise ValueError("factor_covariance must be positive semidefinite")
    C = basis * (np.sqrt(np.maximum(eig, 0)) * np.sqrt(scale))
    U = X @ C
    D = np.asarray(specific_covariance, dtype=float)
    if D.shape == (n, n):
        d = np.diag(D).copy()
        if np.count_nonzero(D) != np.count_nonzero(d):
            raise ValueError("specific_covariance must be diagonal; use the dense API for correlated D")
    else:
        d = _vector(D, n, "specific variances")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("specific variances must be finite and nonnegative (not volatilities)")
    w = _weights(_vector(market_weights, n, "market_weights"), n)

    base_market_covariance = U @ (U.T @ w) + d * w
    s = float(w @ base_market_covariance)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("market variance must be finite and positive")
    beta = base_market_covariance / s
    calibration = calibrate_betas(
        beta, w, lower_bound=lower_bound, tolerance=tolerance, max_iterations=max_iterations,
    )
    delta = calibration.beta - beta
    z_market = w + w * float(delta @ w)
    base = U @ (U.T @ z_market) + d * z_market
    new_market_covariance = base + delta * float(w @ base)
    new_s = float(w @ new_market_covariance)
    if not np.isfinite(new_s) or new_s <= 0:
        raise NumericalError("invalid updated market variance")
    beta_after = new_market_covariance / new_s
    diagnostics = {
        "weighted_mean_error": float(w @ calibration.beta - 1),
        "market_variance_relative_error": abs(new_s / s - 1),
        "beta_reconstruction_max_error": float(np.max(np.abs(beta_after - calibration.beta))),
        "floor_violation": max(0.0, lower_bound - float(beta_after.min())),
        "factor_square_root_relative_error": float(np.max(np.abs(C @ C.T - F))) / scale,
    }
    verification_tol = max(1e-9, 20 * tolerance)
    if (not np.isfinite(U).all() or not np.isfinite(beta_after).all()
            or diagnostics["market_variance_relative_error"] > verification_tol
            or diagnostics["beta_reconstruction_max_error"] > verification_tol * max(1, np.max(np.abs(calibration.beta)))
            or diagnostics["floor_violation"] > verification_tol):
        raise NumericalError(f"structured reconstruction failed: {diagnostics}")
    return StructuredCovariance(
        factor_risk_loadings=_readonly(U), specific_variances=_readonly(d),
        market_weights=_readonly(w), beta_before=_readonly(beta),
        beta_target=_readonly(calibration.beta), beta_after=_readonly(beta_after), delta=_readonly(delta),
        market_variance=s, updated_market_variance=new_s, shift=calibration.shift,
        iterations=calibration.iterations, diagnostics=diagnostics,
    )
