"""Solver-independent adjusted risk payload and pure NumPy calculations."""

import numpy as np

from stock_covariance import StructuredCovariance
from .risk import BarraView
from ._validation import _array


def risk_arrays(risk):
    """Export a full model or subset as independent NumPy arrays and one float.

    Save all six fields together in context; no original model or solver is
    needed afterward. U is risk-scaled loading, NOT the original Barra X.
    """
    if isinstance(risk, StructuredCovariance):
        U, d, w, delta = (risk.factor_risk_loadings, risk.specific_variances,
                          risk.market_weights, risk.delta)
        h, v_out = U.T @ w, 0.
    elif isinstance(risk, BarraView):
        U, d, w, delta, h, v_out = (risk.U, risk.d, risk.w, risk.delta, risk.h,
                                    risk.outside_specific_variance)
    else:
        raise TypeError('risk must be adjusted.risk or adjusted.for_symbols(...)')
    result = {key: np.array(value, dtype=float, copy=True)
              for key, value in dict(U=U, d=d, w=w, delta=delta, h=h).items()}
    for array in result.values():
        array.flags.writeable = False
    return {**result, 'v_out': float(v_out)}


def portfolio_variance(data, exposure):
    """Compute p.T @ Sigma_star @ p from a risk_arrays() payload in O(MK)."""
    p = _array(exposure, (len(data['d']),), 'exposure')
    t = float(data['delta'] @ p)
    z = p + data['w'] * t
    y = data['U'].T @ p + data['h'] * t
    return float(y @ y + data['d'] @ (z*z) + data['v_out']*t*t)


def covariance_matrix(data):
    """Materialize the same adjusted covariance from the numeric payload.

    O(M^2*K) work and O(M^2) output; call only if downstream needs a matrix.
    """
    delta, d, w = data['delta'], data['d'], data['w']
    G = data['U'] + np.outer(delta, data['h'])
    covariance = G @ G.T
    covariance[np.diag_indices_from(covariance)] += d
    dw = d*w
    covariance += np.outer(dw, delta) + np.outer(delta, dw)
    covariance += (float(dw @ w) + data['v_out']) * np.outer(delta, delta)
    return .5*covariance + .5*covariance.T
