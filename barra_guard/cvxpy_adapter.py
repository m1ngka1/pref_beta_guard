"""Optional CVXPY expression builder, independent of preprocessing/context.

Not imported by barra_guard. Ignore this file when using another solver.
"""

import cvxpy as cp
import numpy as np


def risk_expression(data, exposure, *, volatility=False):
    """Return (expression, constraints) from a risk_arrays() numeric payload.

    Add ALL constraints to the downstream problem. This builds expressions;
    it never solves, sets solver options, or stores solver objects in data.
    volatility=False gives variance; True gives its SOC-compatible square root.
    """
    p = cp.Expression.cast_to_const(exposure)
    m, k = data['U'].shape
    if p.shape == (m, 1):
        p = cp.reshape(p, (m,), order='F')
    if p.shape != (m,) or not p.is_affine():
        raise ValueError('exposure must be an affine vector matching the risk data')
    y = cp.Variable(k)
    if np.any(data['delta'] != 0):
        t, z = cp.Variable(), cp.Variable(m)
        constraints = [t == data['delta'] @ p, z == p + data['w']*t,
                       y == data['U'].T @ p + data['h']*t]
    else:
        t, z = cp.Constant(0.), p
        constraints = [y == data['U'].T @ p]
    if volatility:
        value = cp.norm(cp.hstack([y, cp.multiply(np.sqrt(data['d']), z),
                                   np.sqrt(data['v_out'])*t]), 2)
    else:
        value = (cp.sum_squares(y) + cp.sum(cp.multiply(data['d'], cp.square(z)))
                 + data['v_out']*cp.square(t))
    return value, constraints
