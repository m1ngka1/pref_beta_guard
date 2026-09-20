"""Run with: uv run python example.py"""

import json

import numpy as np

from beta_guard import InfeasibleAdjustmentError, adjust_factor_covariance


def main():
    X = np.column_stack([np.ones(4), [-1.1, -0.9, 0.8, 1.2]])
    F = np.array([[1, 1], [1, 2]], dtype=float)
    specific_variances = np.full(4, 0.04)
    weights = np.full(4, 0.25)
    result = adjust_factor_covariance(
        X, F, specific_variances, weights,
        preserve_order=True, pin_negative=True, check_psd=True,
    )
    print(json.dumps({
        "status": result.status,
        "beta_before": result.beta_before.tolist(),
        "beta_target": result.beta_target.tolist(),
        "beta_recomputed": result.beta_after.tolist(),
        "h": result.h.tolist(),
        "factor_covariance": result.factor_covariance.tolist(),
        "market_variance": result.market_variance,
        "diagnostics": result.diagnostics,
    }, indent=2))
    try:
        adjust_factor_covariance([[1], [-1]], [[1]], [0.01, 0.01], [0.75, 0.25])
    except InfeasibleAdjustmentError as exc:
        print(f"Expected infeasible example: {exc}")
    else:
        raise AssertionError("the one-factor example must be infeasible")


if __name__ == "__main__":
    main()
