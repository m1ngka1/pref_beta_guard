"""Run from installed package: python examples/standalone_barra.py."""

import cvxpy as cp
import numpy as np

from barra_guard import BarraSnapshot, adjust_barra


source = BarraSnapshot(
    as_of='2026-06-19', model_id='DEMO', currency='JPY', horizon='daily',
    symbols=('A', 'B', 'C', 'D'), factor_names=('market', 'size'),
    factor_exposure=np.array([[-.4, .3], [.1, -.2], [1.8, .4], [2.3, .1]]),
    factor_covariance=np.diag([.0004, .0001]), specific_variance=np.full(4, .0002),
    market_weights=np.full(4, .25),
)
adjusted = adjust_barra(source)
view = adjusted.for_symbols(['C', 'A', 'B'])  # Full-market calibration; chosen output order.
p = cp.Variable(3)
block = view.cvxpy_risk(p)
problem = cp.Problem(cp.Minimize(block.variance), [p >= 0, cp.sum(p) == 1, *block.constraints])
problem.solve(solver='CLARABEL')
if problem.status != cp.OPTIMAL:
    raise RuntimeError(problem.status)
np.testing.assert_allclose(block.variance.value, p.value @ view.to_dense() @ p.value, atol=1e-10)
print('symbols:', view.symbols)
print('beta*:', view.predicted_beta)
print('weights:', p.value)
print('variance:', view.variance(p.value))
