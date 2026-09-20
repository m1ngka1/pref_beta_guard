"""Shared tolerance and roundoff cleanup for full-universe market weights."""

import numpy as np

DEFAULT_WEIGHT_TOLERANCE = 1e-8


def validate_weight_tolerance(tolerance):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("weight tolerance must be finite and strictly between 0 and 1")


def weight_diagnostics(weights):
    total = float(weights.sum())
    return {
        "weight_sum": total,
        "full_investment_error": abs(total - 1.0),
        "minimum_weight": float(weights.min()),
        "negative_weight_mass": float(-np.minimum(weights, 0).sum()),
    }


def prepare_market_weights(value, n, *, tolerance=DEFAULT_WEIGHT_TOLERANCE):
    """Validate raw weights, then clip tiny negatives and normalize a copy.

    Both abs(sum(w)-1) and TOTAL negative mass must be <= tolerance before
    any repair. This is not a projection of arbitrary weights onto a simplex.
    Returns (cleaned weights, diagnostics); never modifies the caller's input.
    """
    validate_weight_tolerance(tolerance)
    raw = np.asarray(value, dtype=float)
    if raw.shape == (n, 1):
        raw = raw[:, 0]
    if n < 1 or raw.shape != (n,) or not np.isfinite(raw).all():
        raise ValueError("market_weights must be a finite N-vector or Nx1 column")
    stats = weight_diagnostics(raw)
    if (stats["full_investment_error"] > tolerance
            or stats["negative_weight_mass"] > tolerance):
        raise ValueError(
            f"market_weights must be nonnegative and sum to 1 within {tolerance:g}: {stats}"
        )
    weights = np.maximum(raw, 0)
    weights /= weights.sum()
    correction = np.abs(weights - raw)
    return weights, {
        **{f"input_{key}": value for key, value in stats.items()},
        "weight_correction_l1": float(correction.sum()),
        "weight_correction_max": float(correction.max()),
    }
