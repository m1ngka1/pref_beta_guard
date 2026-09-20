import cvxpy as cp
import numpy as np
import pytest

from barra_guard import adjust_barra, portfolio_variance, risk_arrays
from barra_guard.cvxpy_adapter import risk_expression


def inputs(n=18, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); X[:, 0] += .8
    F = np.array([[.04, .002, 0], [.002, .02, .001], [0, .001, .01]])
    return dict(factor_exposure=X, factor_covariance=F,
                specific_variance=rng.uniform(.02, .04, n), market_weights=np.full(n, 1/n),
                symbols=tuple(f'S{i}' for i in range(n)))


@pytest.mark.parametrize('seed', range(8))
def test_compact_subset_is_full_principal_covariance(seed):
    adjusted = adjust_barra(**inputs(seed=seed))
    idx = [9, 2, 14, 0]
    view = adjusted.for_symbols([adjusted.symbols[i] for i in idx])
    dense = adjusted.to_dense()[np.ix_(idx, idx)]
    p = np.array([.4, -.8, .1, .6])
    assert view.outside_specific_variance > 0
    np.testing.assert_allclose(view.to_dense(), dense, atol=1e-12)
    np.testing.assert_allclose(view.variance(p), p @ dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.matvec(p), dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.gradient(p), 2*dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.diagonal(), np.diag(dense), atol=1e-12)
    variable = cp.Variable(4)
    variance, constraints = risk_expression(risk_arrays(view), variable)
    problem = cp.Problem(cp.Minimize(variance), [variable == p, *constraints])
    problem.solve(solver='CLARABEL')
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(variance.value, p @ dense @ p, atol=1e-10)
    # Including only subset idiosyncratic risk would silently lose this term.
    t, z, y = view._parts(p)
    assert abs(view.variance(p) - (y@y + view.d@(z*z)) - view.outside_specific_variance*t*t) < 1e-12


def test_recovery_and_supplied_weights_are_same():
    source = inputs()
    known = adjust_barra(**source)
    recovered = adjust_barra(**{**source, 'market_weights': None, 'predicted_beta': known.risk.beta_before})
    assert recovered.weights_source == 'recovered'
    np.testing.assert_allclose(known.to_dense(), recovered.to_dense(), atol=1e-10)
    assert known.weights_source == 'provided'
    with pytest.raises(ValueError, match='disagrees'):
        adjust_barra(**source, predicted_beta=known.risk.beta_before + .1)
    with pytest.raises(ValueError, match='provide'):
        adjust_barra(**{**source, 'market_weights': None})


def test_subset_solver_size_does_not_grow_with_full_universe():
    sizes = []
    for n in [100, 1000]:
        source = inputs(n=n)
        adjusted = adjust_barra(**source)
        # Include the most adjusted name so no-op cannot remove the transform.
        idx = int(np.argmax(np.abs(adjusted.risk.delta)))
        names = [source['symbols'][idx]] + [s for i, s in enumerate(source['symbols']) if i != idx][:4]
        view = adjusted.for_symbols(names)
        p = cp.Variable(5)
        variance, constraints = risk_expression(risk_arrays(view), p)
        problem = cp.Problem(cp.Minimize(variance), [p >= 0, cp.sum(p) == 1, *constraints])
        data, _, _ = problem.get_problem_data(cp.OSQP)
        sizes.append((data['P'].shape, data['P'].nnz, data['A'].shape, data['A'].nnz))
    assert sizes[0] == sizes[1]


def test_noop_and_full_view():
    source = inputs()
    full = adjust_barra(**source).for_symbols(source['symbols'])
    assert full.outside_specific_variance == 0
    np.testing.assert_allclose(full.variance(np.ones(18)), full.adjusted.risk.variance(np.ones(18)), atol=1e-10)
    source.update(factor_exposure=np.ones((18, 3)), specific_variance=np.ones(18))
    view = adjust_barra(**source).for_symbols(['S0', 'S1'])
    p = cp.Variable(2, value=np.ones(2))
    variance, constraints = risk_expression(risk_arrays(view), p)
    problem = cp.Problem(cp.Minimize(variance), [p == np.ones(2), *constraints])
    problem.solve(solver='CLARABEL')
    np.testing.assert_allclose(variance.value, view.variance(p.value), atol=1e-10)
    assert len(constraints) == 1


def test_prepare_once_reuse_across_future_positions():
    source = inputs()
    adjusted = adjust_barra(**source)
    view = adjusted.for_symbols(['S9', 'S2', 'S0'])
    dense = view.to_dense()
    positions = cp.Variable((3, 3))
    prices = np.array([[10., 20., 30.], [11., 21., 31.], [12., 22., 32.]])
    payload = risk_arrays(view)
    blocks = [risk_expression(payload, cp.multiply(prices[t], positions[t])) for t in range(3)]
    holdings = np.array([[1., 2., 3.], [2., -1., 3.], [4., -3., 1.]])
    for dollars in prices*holdings:
        np.testing.assert_allclose(portfolio_variance(payload, dollars), dollars @ dense @ dollars, atol=1e-10)
    problem = cp.Problem(cp.Minimize(sum(value for value, _ in blocks)),
                         [positions == holdings, *(c for _, constraints in blocks for c in constraints)])
    problem.solve(solver='CLARABEL')
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(problem.value, sum(view.variance(p) for p in prices*holdings), atol=1e-9)
    # Inputs changing later cannot mutate the prepared risk snapshot in context.
    source['factor_exposure'][:] = 99
    source['factor_covariance'][:] = 99
    source['specific_variance'][:] = 99
    source['market_weights'][:] = 99
    np.testing.assert_array_equal(view.to_dense(), dense)


def test_optional_labels_and_diagonal_column_inputs():
    source = inputs()
    expected = adjust_barra(**source)
    source.pop('symbols')
    source['specific_variance'] = np.diag(source['specific_variance'])
    source['market_weights'] = source['market_weights'][:, None]
    actual = adjust_barra(**source, predicted_beta=expected.risk.beta_before[:, None])
    assert actual.symbols is None
    np.testing.assert_allclose(actual.to_dense(), expected.to_dense(), atol=1e-12)
    with pytest.raises(ValueError, match='provide symbols'):
        actual.for_symbols(['S0'])


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'unknown', 'time_axis', 'beta_nan', 'beta_shape', 'bad_w'])
def test_invalid_input(case):
    data = inputs()
    if case == 'duplicate':
        data['symbols'] = ('S0',)*18
    elif case == 'missing':
        data['symbols'] = ('S0',)
    elif case == 'time_axis':
        data['factor_exposure'] = data['factor_exposure'][None, :, :]
    elif case == 'beta_nan':
        data['predicted_beta'] = np.full(18, np.nan)
    elif case == 'beta_shape':
        data['predicted_beta'] = np.ones((2, 18))
    elif case == 'bad_w':
        data['predicted_beta'] = adjust_barra(**data).risk.beta_before
        data['market_weights'][0] = -.1  # Do not recover instead of rejecting supplied bad w.
    with pytest.raises(ValueError):
        adjusted = adjust_barra(**data)
        if case == 'unknown':
            adjusted.for_symbols(['absent'])
