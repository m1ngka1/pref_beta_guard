"""End-to-end checks using factor_covariance's environment, which contains CVXPY.

Run from the project root: factor_covariance/.venv/bin/python verify_optimizer.py
The two implementations themselves have no cross-directory dependencies.
"""

import json
import platform

import cvxpy as cp
import numpy as np

from stock_covariance.beta_guard import adjust_covariance
from factor_covariance.beta_guard import InfeasibleAdjustmentError, adjust_factor_covariance


def optimize(covariance, beta):
    holdings = cp.Variable(len(beta))
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(holdings, covariance)),
        [cp.sum(holdings) == 1, holdings >= 0, holdings <= 0.7,
         beta @ holdings >= 0.3, beta @ holdings <= 1.1],
    )
    # Use the returned covariance directly, without assume_PSD/psd_wrap.
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    assert problem.status == cp.OPTIMAL, problem.status
    p = holdings.value
    assert abs(p.sum() - 1) < 1e-8
    assert p.min() >= -1e-8 and p.max() <= 0.7 + 1e-8
    assert 0.3 - 1e-8 <= beta @ p <= 1.1 + 1e-8
    return {
        "status": problem.status,
        "holdings": p.tolist(),
        "portfolio_beta": float(beta @ p),
        "portfolio_variance": float(p @ covariance @ p),
    }


def main():
    X = np.column_stack([np.ones(4), [-1.1, -0.9, 0.8, 1.2]])
    F = np.array([[1, 1], [1, 2]], dtype=float)
    D = np.full(4, 0.04)
    w = np.full(4, 0.25)
    sigma = X @ F @ X.T + np.diag(D)
    stock_result = adjust_covariance(sigma, w, check_psd=True)
    factor_result = adjust_factor_covariance(X, F, D, w, preserve_order=True, check_psd=True)
    report = {"python": platform.python_version(), "numpy": np.__version__, "cvxpy": cp.__version__}
    for name, result in [("stock_covariance", stock_result), ("factor_covariance", factor_result)]:
        assert result.beta_after.min() >= 0.15 - 1e-7
        report[name] = {
            "beta_before": result.beta_before.tolist(),
            "beta_after": result.beta_after.tolist(),
            "min_covariance_eigenvalue": float(np.linalg.eigvalsh(result.covariance).min()),
            "diagnostics": result.diagnostics,
            "downstream_optimization": optimize(result.covariance, result.beta_after),
        }

    # Same valid model: the factor-restricted problem is impossible, while
    # the name-by-name construction still gives a valid adjusted covariance.
    X_bad = np.array([[1.0], [-1.0]])
    w_bad = np.array([0.75, 0.25])
    sigma_bad = X_bad @ X_bad.T + 0.01 * np.eye(2)
    try:
        adjust_factor_covariance(X_bad, [[1]], [0.01, 0.01], w_bad)
    except InfeasibleAdjustmentError:
        report["restricted_case_factor"] = "infeasible (expected)"
    else:
        raise AssertionError("expected factor-covariance infeasibility")
    repaired = adjust_covariance(sigma_bad, w_bad, check_psd=True)
    assert repaired.beta_after.min() >= 0.15 - 1e-9
    report["restricted_case_stock"] = repaired.beta_after.tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
