"""Exact selected-name risk calculations; no planner or date handling."""

import numpy as np

from ._validation import _array


class BarraView:
    """Exact principal risk for M selected names, with only O(MK+M) numeric data.

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
