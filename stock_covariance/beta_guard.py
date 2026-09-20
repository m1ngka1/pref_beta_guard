"""Monotone beta floor + closed-form stock covariance update; NumPy only."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from market_weights import DEFAULT_WEIGHT_TOLERANCE, prepare_market_weights

FloatArray = NDArray[np.float64]


class NumericalError(RuntimeError):
    """Calibration or reconstructed covariance failed the numerical checks."""


@dataclass(frozen=True)
class BetaCalibration:
    beta: FloatArray
    shift: float
    iterations: int
    weighted_mean_error: float


@dataclass(frozen=True)
class AdjustmentResult:
    covariance: FloatArray
    market_weights: FloatArray
    beta_before: FloatArray
    beta_target: FloatArray
    beta_after: FloatArray  # Recomputed from the returned covariance.
    market_variance: float
    shift: float
    iterations: int
    diagnostics: dict[str, float]


def _vector(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty, finite 1-D array")
    return array


def _covariance(value: ArrayLike, check_psd: bool) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()):
        raise ValueError("covariance must be a nonempty, finite square matrix")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > 1e-10 * scale:
        raise ValueError("covariance must be symmetric")
    matrix = 0.5 * matrix + 0.5 * matrix.T
    if check_psd and np.linalg.eigvalsh(matrix / scale)[0] < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def calibrate_betas(
    beta: ArrayLike,
    market_weights: ArrayLike,
    *,
    lower_bound: float = 0.15,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    tolerance: float = 1e-12,
    max_iterations: int = 128,
) -> BetaCalibration:
    """Solve w @ maximum(lower_bound, beta - shift) == 1 by bisection.

    Requires long-only, fully invested market weights and original w @ beta == 1.
    Weight errors within weight_tolerance are cleaned; beta must remain
    consistent with the cleaned weights. The output preserves weak ordering,
    including for zero-weight assets.
    Raises NumericalError on nonconvergence; never returns a partial iterate.
    """
    beta = _vector(beta, "beta")
    weights, _ = prepare_market_weights(market_weights, beta.size, tolerance=weight_tolerance)
    if not np.isfinite(lower_bound) or not 0 < lower_bound < 1:
        raise ValueError("lower_bound must be strictly between 0 and 1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if (isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    if abs(float(weights @ beta) - 1.0) > 1e-10:
        raise ValueError("original market-weighted beta must equal 1")

    clipped = np.maximum(lower_bound, beta)
    error = float(weights @ clipped - 1.0)
    if abs(error) <= tolerance:
        return BetaCalibration(clipped, 0.0, 0, error)
    if error < 0:
        raise NumericalError("original beta is not calibrated closely enough for this tolerance")

    lo = 0.0
    # Zero-weight outliers do not need to enlarge the root-search interval.
    hi = float(np.max(beta[weights > 0]) - lower_bound)
    for iteration in range(1, max_iterations + 1):
        shift = lo + (hi - lo) / 2
        target = np.maximum(lower_bound, beta - shift)
        error = float(weights @ target - 1.0)
        if abs(error) <= tolerance:
            return BetaCalibration(target, shift, iteration, error)
        if shift == lo or shift == hi:
            break
        if error > 0:
            lo = shift
        else:
            hi = shift
    raise NumericalError(
        f"bisection did not reach tolerance {tolerance:g}; last mean error={error:.3g}"
    )


def adjust_covariance(
    covariance: ArrayLike,
    market_weights: ArrayLike,
    *,
    lower_bound: float = 0.15,
    weight_tolerance: float = DEFAULT_WEIGHT_TOLERANCE,
    tolerance: float = 1e-12,
    max_iterations: int = 128,
    check_psd: bool = False,
) -> AdjustmentResult:
    """Adjust a full stock covariance while preserving market variance.

    Input covariance must be PSD. check_psd=True adds an O(N^3) eigenvalue
    validation; it is off by default for trusted risk-model inputs. No PSD
    repair or covariance-unit conversion is performed. Weight errors within
    weight_tolerance are clipped/normalized before computing beta and variance.
    Matrix reconstruction costs O(N^2); each bisection iteration costs O(N).
    All inputs are left unchanged. beta_after is recomputed, not just the target.
    """
    sigma = _covariance(covariance, check_psd)
    weights, weight_stats = prepare_market_weights(market_weights, sigma.shape[0], tolerance=weight_tolerance)
    market_covariance = sigma @ weights
    variance = float(weights @ market_covariance)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("market variance must be finite and strictly positive")
    beta = market_covariance / variance
    calibration = calibrate_betas(
        beta, weights, lower_bound=lower_bound, tolerance=tolerance,
        max_iterations=max_iterations, weight_tolerance=weight_tolerance,
    )
    delta = calibration.beta - beta
    # Expansion of s*(beta_new beta_new' - beta beta') avoids subtracting
    # two nearly equal outer products when the change is small.
    updated = sigma + variance * (
        np.outer(beta, delta) + np.outer(delta, beta) + np.outer(delta, delta)
    )
    updated = 0.5 * updated + 0.5 * updated.T
    if not np.isfinite(updated).all():
        raise NumericalError("nonfinite covariance after update")
    new_variance = float(weights @ updated @ weights)
    if not np.isfinite(new_variance) or new_variance <= 0:
        raise NumericalError("nonpositive market variance after update")
    realized = updated @ weights / new_variance
    diagnostics = {
        "weighted_mean_error": float(weights @ calibration.beta - 1.0),
        "market_variance_relative_error": abs(new_variance / variance - 1.0),
        "beta_reconstruction_max_error": float(np.max(np.abs(realized - calibration.beta))),
        "floor_violation": max(0.0, lower_bound - float(np.min(realized))),
    }
    verification_tol = max(1e-9, 20 * tolerance)
    beta_scale = max(1.0, float(np.max(np.abs(calibration.beta))))
    if (not np.isfinite(realized).all()
            or diagnostics["market_variance_relative_error"] > verification_tol
            or diagnostics["beta_reconstruction_max_error"] > verification_tol * beta_scale
            or diagnostics["floor_violation"] > verification_tol):
        raise NumericalError(f"covariance reconstruction failed validation: {diagnostics}")
    if check_psd:
        try:
            _covariance(updated, True)
        except ValueError as exc:
            raise NumericalError("updated covariance failed the PSD check") from exc
    diagnostics.update(weight_stats)
    return AdjustmentResult(
        covariance=updated, market_weights=weights, beta_before=beta, beta_target=calibration.beta,
        beta_after=realized, market_variance=variance, shift=calibration.shift,
        iterations=calibration.iterations, diagnostics=diagnostics,
    )
