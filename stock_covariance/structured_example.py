"""Standalone integration example: uv run --extra optimization python structured_example.py."""

import json

import cvxpy as cp
import numpy as np

from structured import adjust_from_factors


def main():
    X = np.array([[-.2, 1.], [.1, -.4], [1.8, .2], [2.3, -.1]])
    F = np.array([[.09, .005], [.005, .02]])
    d = np.full(4, .04)  # Variances, NOT standard deviations.
    w = np.full(4, .25)
    model = adjust_from_factors(X, F, d, w, lower_bound=.15)

    p = cp.Variable(4, name='holdings')
    alpha = cp.Parameter(4, value=np.array([.01, .015, .02, .025]))
    risk = model.cvxpy_risk(p)
    problem = cp.Problem(cp.Minimize(3 * risk.variance - alpha @ p), [
        p >= 0, cp.sum(p) == 1, p <= .7,
        model.beta_after @ p >= .3, model.beta_after @ p <= 1.1,
        *risk.constraints,  # REQUIRED: these make risk equal p.T @ Sigma_star @ p.
    ])
    problem.solve(solver='OSQP', eps_abs=1e-9, eps_rel=1e-9)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(problem.status)
    matrix = model.to_dense()  # Optional export, unnecessary for the optimization above.
    np.testing.assert_allclose(risk.variance.value, p.value @ matrix @ p.value, atol=1e-9)
    print(json.dumps({
        'beta_before': model.beta_before.tolist(), 'beta_after': model.beta_after.tolist(),
        'holdings': p.value.tolist(), 'variance': model.variance(p.value),
        'gradient': model.gradient(p.value).tolist(), 'status': problem.status,
        'diagnostics': model.diagnostics,
    }, indent=2))


if __name__ == '__main__':
    main()
