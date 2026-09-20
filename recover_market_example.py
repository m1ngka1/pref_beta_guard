"""Synthetic recovery with X (NxK), dense F (KxK), diagonal D and column beta.

Run from the project root:
    stock_covariance/.venv/bin/python recover_market_example.py
"""

import json

import numpy as np

from market_recovery import recover_from_covariance, recover_from_factors


def main():
    X = np.column_stack([np.ones(6), [-1.2, -0.8, 0.0, 0.4, 1.0, 1.5]])
    F = np.array([[0.04, 0.035], [0.035, 0.09]])
    D = np.diag([0.03, 0.02, 0.04, 0.015, 0.025, 0.035])

    # Known ONLY to generate/check the example; neither recovery function
    # receives these weights or the market variance.
    true_weights = np.array([0.10, 0.20, 0.25, 0.15, 0.20, 0.10])
    sigma = X @ F @ X.T + D
    true_variance = float(true_weights @ sigma @ true_weights)
    beta = (sigma @ true_weights / true_variance)[:, None]

    recovered = recover_from_factors(X, F, D, beta)
    dense = recover_from_covariance(sigma, beta)
    np.testing.assert_allclose(recovered.weights, true_weights, atol=1e-12)
    np.testing.assert_allclose(recovered.weights, dense.weights, atol=1e-12)
    np.testing.assert_allclose(recovered.market_variance, true_variance, rtol=1e-12)
    assert recovered.market_consistent
    print(json.dumps({
        "data": "synthetic",
        "input_shapes": {"X": list(X.shape), "F": list(F.shape),
                         "D": list(D.shape), "beta": list(beta.shape)},
        "original_beta": beta[:, 0].tolist(),
        "true_weights": true_weights.tolist(),
        "recovered_weights": recovered.weights.tolist(),
        "true_market_variance": true_variance,
        "recovered_market_variance": recovered.market_variance,
        "recovered_market_volatility": recovered.market_volatility,
        "max_weight_error": float(np.max(np.abs(recovered.weights - true_weights))),
        "dense_vs_factor_weight_error": float(np.max(np.abs(recovered.weights - dense.weights))),
        "market_consistent": recovered.market_consistent,
        "diagnostics": recovered.diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
