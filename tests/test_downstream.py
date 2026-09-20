"""Both covariance methods feed a constrained optimizer without PSD bypasses."""

import cvxpy as cp
import numpy as np
import pytest

from factor_covariance import adjust_factor_covariance
from stock_covariance import adjust_covariance


@pytest.mark.parametrize('method', ['stock', 'factor'])
def test_adjusted_covariance_in_portfolio_optimizer(method):
    X = np.column_stack([np.ones(4), [-1.1, -.9, .8, 1.2]])
    F, d, w = np.array([[1., 1.], [1., 2.]]), np.full(4, .04), np.full(4, .25)
    if method == 'stock':
        result = adjust_covariance(X @ F @ X.T + np.diag(d), w, check_psd=True)
    else:
        result = adjust_factor_covariance(X, F, d, w, preserve_order=True, check_psd=True)
    p = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.quad_form(p, result.covariance)), [
        p >= 0, cp.sum(p) == 1, p <= .7,
        result.beta_after @ p >= .3, result.beta_after @ p <= 1.1,
    ])
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    assert problem.status == cp.OPTIMAL
    assert abs(p.value.sum() - 1) < 1e-8
    assert p.value.min() >= -1e-8 and p.value.max() <= .7 + 1e-8
    assert .3 - 1e-8 <= result.beta_after @ p.value <= 1.1 + 1e-8
    np.testing.assert_allclose(problem.value, p.value @ result.covariance @ p.value, atol=1e-10)
