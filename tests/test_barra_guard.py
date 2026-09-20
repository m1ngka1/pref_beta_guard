from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from barra_guard import BarraSnapshot, adjust_barra
from barra_guard.planner import DensePlannerRiskAdapter, PlannerRiskAdapter
from barra_guard.tabular import adjust_factor_risk_data


def snapshot(n=18, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)); X[:, 0] += .8
    F = np.array([[.04, .002, 0], [.002, .02, .001], [0, .001, .01]])
    d, w = rng.uniform(.02, .04, n), np.full(n, 1/n)
    return BarraSnapshot('2026-06-19', 'SYNTHETIC', 'JPY', 'daily',
                         tuple(f'S{i}' for i in range(n)), ('market', 'size', 'value'), X, F, d, w)


@pytest.mark.parametrize('seed', range(8))
def test_compact_subset_is_full_principal_covariance(seed):
    adjusted = adjust_barra(snapshot(seed=seed))
    idx = [9, 2, 14, 0]
    view = adjusted.for_symbols([adjusted.source.symbols[i] for i in idx])
    dense = adjusted.to_dense()[np.ix_(idx, idx)]
    p = np.array([.4, -.8, .1, .6])
    assert view.outside_specific_variance > 0
    np.testing.assert_allclose(view.to_dense(), dense, atol=1e-12)
    np.testing.assert_allclose(view.variance(p), p @ dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.matvec(p), dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.gradient(p), 2*dense @ p, atol=1e-12)
    np.testing.assert_allclose(view.diagonal(), np.diag(dense), atol=1e-12)
    variable = cp.Variable(4)
    block = view.cvxpy_risk(variable)
    problem = cp.Problem(cp.Minimize(block.variance), [variable == p, *block.constraints])
    problem.solve(solver='CLARABEL')
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(block.variance.value, p @ dense @ p, atol=1e-10)
    # Including only subset idiosyncratic risk would silently lose this term.
    t, z, y = view._parts(p)
    assert abs(view.variance(p) - (y@y + view.d@(z*z)) - view.outside_specific_variance*t*t) < 1e-12


def test_recovery_and_supplied_weights_are_same():
    source = snapshot()
    known = adjust_barra(source)
    recovered = adjust_barra(replace(source, market_weights=None, predicted_beta=known.risk.beta_before))
    assert recovered.weights_source == 'recovered'
    np.testing.assert_allclose(known.to_dense(), recovered.to_dense(), atol=1e-10)
    assert known.weights_source == 'provided'
    with pytest.raises(ValueError, match='disagrees'):
        adjust_barra(replace(source, predicted_beta=known.risk.beta_before + .1))
    with pytest.raises(ValueError, match='provide'):
        adjust_barra(replace(source, market_weights=None))


def test_snapshots_and_bad_labels():
    source = snapshot()
    with pytest.raises(ValueError):
        source.factor_exposure[0, 0] = 8
    with pytest.raises(ValueError, match='unique'):
        replace(source, symbols=('same',)*18)
    with pytest.raises(ValueError, match='absent'):
        adjust_barra(source).for_symbols(['missing'])
    with pytest.raises(ValueError, match='unique'):
        adjust_barra(source).for_symbols(['S0', 'S0'])
    with pytest.raises(ValueError):
        replace(source, specific_variance=np.diag(source.specific_variance))


def frames():
    source = snapshot()
    X = pd.DataFrame(source.factor_exposure, index=source.symbols, columns=source.factor_names)
    F = pd.DataFrame(source.factor_covariance, index=source.factor_names, columns=source.factor_names)
    d = pd.Series(source.specific_variance, index=source.symbols)
    w = pd.Series(source.market_weights, index=source.symbols)
    return source, SimpleNamespace(factor_exposure=X, factor_covariance=F.iloc[::-1, ::-1], specific_variance=d.iloc[::-1]), w.iloc[::-1]


def prepare(data, **kwargs):
    return adjust_factor_risk_data(data, dates=['2026-06-19', '2026-06-22'], model_id='SYNTHETIC',
                                   currency='JPY', horizon='daily', **kwargs)


@pytest.mark.parametrize('form', ['static', 'panel', 'mapping', 'arrays', 'cov_panel'])
def test_trade_planner_data_shapes_and_alignment(form):
    source, data, w = frames()
    dates = pd.to_datetime(['2026-06-19', '2026-06-22'])
    if form in ('panel', 'mapping', 'cov_panel'):
        data.factor_exposure = pd.concat([data.factor_exposure, data.factor_exposure.iloc[::-1]], keys=dates)
        data.specific_variance = pd.concat([data.specific_variance, data.specific_variance], keys=dates).to_frame('specific_variance')
        w = pd.DataFrame([w, w], index=dates)
    if form == 'mapping':
        data.factor_covariance = {str(d): data.factor_covariance for d in dates}
    if form == 'cov_panel':
        data.factor_covariance = pd.concat([data.factor_covariance]*2, keys=dates)
    if form == 'arrays':
        data.factor_covariance = np.stack([source.factor_covariance]*2)
        data.specific_variance = np.stack([source.specific_variance]*2)
        w = source.market_weights
    panel = prepare(data, market_weights=w)
    expected = adjust_barra(source)
    for day in dates:
        np.testing.assert_allclose(panel.at(day).to_dense(), expected.to_dense(), atol=1e-12)
    assert panel.beta_frame(['S7', 'S1']).shape == (2, 2)
    assert tuple(panel.beta_frame().columns) == source.symbols


def test_labeled_beta_recovery():
    source, data, _ = frames()
    beta = pd.Series(adjust_barra(source).risk.beta_before, index=source.symbols).iloc[::-1]
    panel = prepare(data, predicted_beta=beta)
    np.testing.assert_allclose(panel.at('2026-06-19').market_weights, source.market_weights, atol=1e-12)


def test_distinct_dates_are_not_forward_filled_or_swapped():
    source, data, w = frames()
    dates = pd.to_datetime(['2026-06-19', '2026-06-22'])
    second = replace(source, as_of='2026-06-22', factor_exposure=source.factor_exposure*1.1,
                     factor_covariance=source.factor_covariance*1.4, specific_variance=source.specific_variance*.7,
                     market_weights=np.arange(1., 19.) / np.arange(1., 19.).sum())
    second_X = pd.DataFrame(second.factor_exposure, index=second.symbols, columns=second.factor_names).iloc[::-1]
    data.factor_exposure = pd.concat([data.factor_exposure, second_X], keys=dates)
    data.factor_covariance = {dates[0]: source.factor_covariance, dates[1]: second.factor_covariance}
    data.specific_variance = np.stack([source.specific_variance, second.specific_variance])
    panel = prepare(data, market_weights=np.stack([source.market_weights, second.market_weights]))
    for expected in [source, second]:
        np.testing.assert_allclose(panel.at(expected.as_of).to_dense(), adjust_barra(expected).to_dense(), atol=1e-12)
    with pytest.raises(ValueError, match='missing date'):
        panel.at('2026-06-20')


@pytest.mark.parametrize('case', ['missing_symbol', 'duplicate', 'missing_date', 'bad_factor', 'nan', 'duplicate_day'])
def test_tabular_rejects_incomplete_or_ambiguous_data(case):
    source, data, w = frames()
    if case == 'missing_symbol':
        data.specific_variance = data.specific_variance.iloc[1:]
    elif case == 'duplicate':
        data.factor_exposure = pd.concat([data.factor_exposure, data.factor_exposure.iloc[:1]])
    elif case == 'missing_date':
        data.specific_variance = pd.DataFrame([data.specific_variance], index=['2026-06-19'])
    elif case == 'bad_factor':
        data.factor_covariance = data.factor_covariance.rename(columns={'market': 'missing'})
    elif case == 'nan':
        w.iloc[0] = np.nan
    else:
        data.factor_covariance = {'2026-06-19': data.factor_covariance, '2026-06-19 12:00': data.factor_covariance}
    with pytest.raises(ValueError):
        prepare(data, market_weights=w)


def test_reference_objective_scaling_and_planner_bridges():
    _, data, w = frames()
    panel = prepare(data, market_weights=w)
    ctx = SimpleNamespace(symbols=['S9', 'S2', 'S0'], dates=pd.to_datetime(panel.dates),
                          price=np.array([[10., 20., 30.], [11., 21., 31.]]))
    adapter, dense = PlannerRiskAdapter(panel), DensePlannerRiskAdapter(panel)
    p = cp.Variable(3)
    block = adapter.risk_block(p, ctx, 0)
    with pytest.raises(ValueError, match='variable value'):
        block.set_reference_values()
    with pytest.raises(TypeError, match='risk_block'):
        adapter.objective(p, ctx, 0)
    for shares in [np.array([2., -3., 4.]), np.array([-1., 2., 3.])]:
        p.value = shares
        block.set_reference_values()
        np.testing.assert_allclose(block.variance.value, dense.objective(p, ctx, 0).value, atol=1e-10)
        np.testing.assert_allclose(block.variance.value, adapter.variance(shares, ctx, 0), atol=1e-10)
    # Trade Planner's per-name scaling: adjusted price times parent units = dollars.
    scale = np.array([10., 30., 50.])
    scaled_ctx = SimpleNamespace(**{**vars(ctx), 'price': ctx.price * scale})
    np.testing.assert_allclose(adapter.variance(p.value/scale, scaled_ctx, 0), adapter.variance(p.value, ctx, 0))


def test_subset_solver_size_does_not_grow_with_full_universe():
    sizes = []
    for n in [100, 1000]:
        source = snapshot(n=n)
        adjusted = adjust_barra(source)
        # Include the most adjusted name so no-op cannot remove the transform.
        idx = int(np.argmax(np.abs(adjusted.risk.delta)))
        names = [source.symbols[idx]] + [s for i, s in enumerate(source.symbols) if i != idx][:4]
        view = adjusted.for_symbols(names)
        p = cp.Variable(5)
        block = view.cvxpy_risk(p)
        problem = cp.Problem(cp.Minimize(block.variance), [p >= 0, cp.sum(p) == 1, *block.constraints])
        data, _, _ = problem.get_problem_data(cp.OSQP)
        sizes.append((data['P'].shape, data['P'].nnz, data['A'].shape, data['A'].nnz))
    assert sizes[0] == sizes[1]


def test_noop_and_full_view():
    source = snapshot()
    full = adjust_barra(source).for_symbols(source.symbols)
    assert full.outside_specific_variance == 0
    np.testing.assert_allclose(full.variance(np.ones(18)), full.adjusted.risk.variance(np.ones(18)), atol=1e-10)
    source = replace(source, factor_exposure=np.ones((18, 3)), specific_variance=np.ones(18))
    view = adjust_barra(source).for_symbols(['S0', 'S1'])
    p = cp.Variable(2, value=np.ones(2))
    block = view.cvxpy_risk(p)
    block.set_reference_values()
    np.testing.assert_allclose(block.variance.value, view.variance(p.value))
    assert len(block.constraints) == 1
