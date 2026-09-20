"""Run with: uv run python example.py"""

import json

import numpy as np

from beta_guard import adjust_covariance


def main():
    weights = np.full(4, 0.25)
    beta = np.array([-0.1, 0.1, 1.8, 2.2])
    residual = 0.04 * (np.eye(4) - np.ones((4, 4)) / 4)
    covariance = 0.09 * np.outer(beta, beta) + residual
    result = adjust_covariance(covariance, weights, check_psd=True)
    print(json.dumps({
        "beta_before": result.beta_before.tolist(),
        "beta_target": result.beta_target.tolist(),
        "beta_recomputed": result.beta_after.tolist(),
        "shift": result.shift,
        "bisection_iterations": result.iterations,
        "market_variance": result.market_variance,
        "min_eigenvalue": float(np.linalg.eigvalsh(result.covariance).min()),
        "diagnostics": result.diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
