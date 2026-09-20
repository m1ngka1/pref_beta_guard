import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from stock_covariance import adjust_from_factors
from barra_guard import risk_arrays
from barra_guard.cvxpy_adapter import risk_expression


def model_input(seed=0, n=30, k=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    X[:, 0] += .8
    R = rng.normal(size=(k, k))
    return X, .04 * (R @ R.T / k + np.eye(k)), rng.uniform(.03, .08, n), np.full(n, 1/n)


@pytest.mark.parametrize('seed', range(6))
@pytest.mark.parametrize('active', [False, True])
def test_same_constrained_optimizer(seed, active):
    model = adjust_from_factors(*model_input(seed))
    n = model.n_assets
    rng = np.random.default_rng(seed + 100)
    alpha = rng.normal(0, .02, n)
    benchmark = np.full(n, 1/n) if active else np.zeros(n)
    previous = np.full(n, 1/n)
    solutions = []
    for structured in [False, True]:
        p = cp.Variable(n)
        constraints = [p >= 0, cp.sum(p) == 1, p <= 4/n,
                       model.beta_after @ p >= .4, model.beta_after @ p <= 1.2,
                       cp.norm1(p - previous) <= .8]
        if structured:
            variance, risk_constraints = risk_expression(risk_arrays(model), p - benchmark)
            constraints.extend(risk_constraints)
        else:
            variance = cp.quad_form(p - benchmark, model.to_dense())
        problem = cp.Problem(cp.Minimize(3 * variance - alpha @ p), constraints)
        problem.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
        assert problem.status == cp.OPTIMAL
        solutions.append((p.value, problem.value))
        np.testing.assert_allclose(variance.value, model.variance(p.value - benchmark), atol=2e-9)
    np.testing.assert_allclose(solutions[0][0], solutions[1][0], atol=3e-5)
    np.testing.assert_allclose(solutions[0][1], solutions[1][1], atol=2e-9)


def test_soc_risk_limit_and_column_vector():
    model = adjust_from_factors(*model_input())
    n = model.n_assets
    p = cp.Variable((n, 1))
    b = np.full((n, 1), 1/n)
    volatility, risk_constraints = risk_expression(risk_arrays(model), p - b, volatility=True)
    problem = cp.Problem(cp.Maximize(np.linspace(0, .02, n) @ p[:, 0]),
                         [p >= 0, cp.sum(p) == 1, volatility <= .08, *risk_constraints])
    problem.solve(solver='CLARABEL')
    assert problem.status == cp.OPTIMAL
    assert np.sqrt(model.variance(p.value - b)) <= .08 + 1e-7


def test_subset_scatter_and_parameter_reuse():
    model = adjust_from_factors(*model_input())
    idx = np.array([0, 4, 9, 16, 29])
    E = sparse.csc_matrix((np.ones(5), (idx, np.arange(5))), shape=(model.n_assets, 5))
    p = cp.Variable(5)
    alpha = cp.Parameter(5)
    variance, risk_constraints = risk_expression(risk_arrays(model), E @ p)
    problem = cp.Problem(cp.Minimize(variance - alpha @ p),
                         [cp.sum(p) == 1, p >= 0, *risk_constraints])
    assert problem.is_dpp()
    for values in [np.zeros(5), np.linspace(0, .02, 5)]:
        alpha.value = values
        problem.solve(solver='OSQP', eps_abs=1e-9, eps_rel=1e-9)
        assert problem.status == cp.OPTIMAL
        np.testing.assert_allclose(variance.value, p.value @ model.to_dense(idx) @ p.value, atol=1e-9)


def test_solver_matrices_keep_sparse_structure():
    n, k = 160, 7
    model = adjust_from_factors(*model_input(n=n, k=k))
    assert np.any(model.delta)
    p = cp.Variable(n)
    variance, risk_constraints = risk_expression(risk_arrays(model), p)
    problem = cp.Problem(cp.Minimize(variance), [cp.sum(p) == 1, p >= 0, *risk_constraints])
    data, _, _ = problem.get_problem_data(cp.OSQP)
    P, A = data['P'], data['A']
    assert (P - sparse.diags(P.diagonal())).nnz == 0
    assert P.nnz <= n + k
    assert A.nnz <= n*k + 6*n + 3*k + 3
    assert data['F'].nnz == n  # Holding inequalities, not transformed exposures.
    assert P.shape[0] <= 3*n + 2*k + 2
    # Catch the tempting but dense inline form. The explicit z equality is essential.
    z_inline = p + model.market_weights * (model.delta @ p)
    inline = cp.Problem(cp.Minimize(cp.sum_squares(model.factor_risk_loadings.T @ z_inline)
                                   + cp.sum_squares(cp.multiply(np.sqrt(model.specific_variances), z_inline))))
    inline_data, _, _ = inline.get_problem_data(cp.OSQP)
    assert inline_data['A'].nnz > n*n


def test_noop_skips_transform_variables():
    model = adjust_from_factors(np.ones((5, 1)), np.ones((1, 1)), np.ones(5), np.full(5, .2))
    p = cp.Variable(5)
    variance, risk_constraints = risk_expression(risk_arrays(model), p)
    assert len(variance.variables()) == 2  # Only p and factor scores.
    assert len(risk_constraints) == 1


def test_reject_nonaffine_and_wrong_size():
    model = adjust_from_factors(*model_input())
    with pytest.raises(ValueError):
        risk_expression(risk_arrays(model), cp.square(cp.Variable(model.n_assets)))
    with pytest.raises(ValueError):
        risk_expression(risk_arrays(model), cp.Variable(model.n_assets - 1))
