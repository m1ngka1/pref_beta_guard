"""Exact selected-name risk calculations; no planner or date handling."""

from dataclasses import dataclass

import numpy as np

from stock_covariance import CvxpyRisk
from ._validation import _array


@dataclass(frozen=True)
class BarraRiskBlock(CvxpyRisk):
    view: 'BarraView'
    exposure: object

    def set_reference_values(self):
        """Populate auxiliaries after setting holdings.value, before evaluating
        a reference objective for numerical scaling. Call anew if holdings change.
        This sets initial values only; it does not replace the equalities.
        """
        if self.exposure.value is None:
            raise ValueError('set the underlying position variable value first')
        t, z, y = self.view._parts(self.exposure.value)
        if np.any(self.view.delta != 0):
            self.shift_exposure.value = t
            self.transformed_exposure.value = z
        self.factor_scores.value = y


class BarraView:
    """Exact principal risk for M selected names, with only O(MK+M) CVXPY data.

    Non-held names influence the market adjustment via h=U.T@w and v_out.
    Never normalize selected market weights or recalibrate on the subset.
    """

    def __init__(self, adjusted, symbols, indices):
        self.adjusted, self.symbols, self.indices = adjusted, symbols, indices
        risk = adjusted.risk
        idx = np.asarray(indices)
        self.U = _array(risk.factor_risk_loadings[idx], (len(idx), risk.n_factors), 'U')
        for name, source in [('d', risk.specific_variances), ('w', risk.market_weights),
                             ('delta', risk.delta), ('predicted_beta', risk.beta_after)]:
            setattr(self, name, _array(source[idx], (len(idx),), name))
        self.h = _array(risk.factor_risk_loadings.T @ risk.market_weights, (risk.n_factors,), 'h')
        outside = np.ones(risk.n_assets, dtype=bool)
        outside[idx] = False
        # Sum outside directly, avoiding catastrophic cancellation total - subset.
        self.outside_specific_variance = float(
            risk.specific_variances[outside] @ (risk.market_weights[outside] ** 2))

    def _parts(self, exposure):
        p = _array(exposure, (len(self.symbols),), 'exposure')
        t = float(self.delta @ p)
        z = p + self.w * t
        y = self.U.T @ p + self.h * t
        return t, z, y

    def variance(self, exposure):
        t, z, y = self._parts(exposure)
        return float(y @ y + self.d @ (z*z) + self.outside_specific_variance*t*t)

    def matvec(self, exposure):
        t, z, y = self._parts(exposure)
        dz = self.d * z
        return self.U @ y + dz + self.delta * (
            self.h @ y + self.w @ dz + self.outside_specific_variance*t)

    def gradient(self, exposure):
        return 2 * self.matvec(exposure)

    def diagonal(self):
        risk = self.adjusted.risk
        beta = risk.beta_before[list(self.indices)]
        return np.einsum('ij,ij->i', self.U, self.U) + self.d + risk.market_variance * (
            2 * beta * self.delta + self.delta**2)

    def to_dense(self):
        return self.adjusted.risk.to_dense(self.indices)

    def cvxpy_risk(self, exposure, *, name='barra'):
        import cvxpy as cp
        p = cp.Expression.cast_to_const(exposure)
        m, k = self.U.shape
        if p.shape == (m, 1):
            p = cp.reshape(p, (m,), order='F')
        if p.shape != (m,) or not p.is_affine():
            raise ValueError('exposure must be an affine vector in view.symbols order')
        y = cp.Variable(k, name=f'{name}_factors')
        if np.any(self.delta != 0):
            t, z = cp.Variable(name=f'{name}_shift'), cp.Variable(m, name=f'{name}_specific')
            constraints = (t == self.delta @ p, z == p + self.w*t, y == self.U.T @ p + self.h*t)
        else:
            t, z = cp.Constant(0.), p
            constraints = (y == self.U.T @ p,)
        variance = (cp.sum_squares(y) + cp.sum(cp.multiply(self.d, cp.square(z)))
                    + self.outside_specific_variance * cp.square(t))
        volatility = cp.norm(cp.hstack([y, cp.multiply(np.sqrt(self.d), z),
                                        np.sqrt(self.outside_specific_variance)*t]), 2)
        return BarraRiskBlock(variance, volatility, constraints, z, y, t, self, p)
