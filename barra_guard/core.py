"""One complete, labeled risk snapshot in; an adjusted risk object out.

The source X/F/d retain their original meaning. Adjusted risk MUST use risk or
for_symbols(); sending just source X/F/d downstream would undo the adjustment.
"""

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np

from market_recovery import recover_from_factors
from stock_covariance import CvxpyRisk, StructuredCovariance, adjust_from_factors


def _labels(values, name):
    result = tuple(values)
    if (not result or any(not isinstance(x, str) or not x for x in result)
            or len(set(result)) != len(result)):
        raise ValueError(f'{name} must contain unique nonempty string labels')
    return result


def _array(value, shape, name):
    result = np.array(value, dtype=float, copy=True)
    if len(shape) == 1 and result.shape == (shape[0], 1):
        result = result[:, 0].copy()
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BarraSnapshot:
    """Aligned single-date data. Covariances use decimal return units.

    as_of is an ISO date. horizon/currency/model_id are explicit provenance,
    not requests for conversion. Arrays are owned read-only snapshots.
    """

    as_of: str
    model_id: str
    currency: str
    horizon: str
    symbols: tuple[str, ...]
    factor_names: tuple[str, ...]
    factor_exposure: np.ndarray
    factor_covariance: np.ndarray
    specific_variance: np.ndarray
    market_weights: np.ndarray | None = None
    predicted_beta: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.as_of, str) or date.fromisoformat(self.as_of).isoformat() != self.as_of:
            raise ValueError('as_of must be an ISO YYYY-MM-DD date')
        for name in ('model_id', 'currency', 'horizon'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('symbols', 'factor_names'):
            object.__setattr__(self, name, _labels(getattr(self, name), name))
        n, k = len(self.symbols), len(self.factor_names)
        for name, shape in [('factor_exposure', (n, k)), ('factor_covariance', (k, k)),
                            ('specific_variance', (n,)), ('market_weights', (n,)), ('predicted_beta', (n,))]:
            value = getattr(self, name)
            if value is None and name in ('market_weights', 'predicted_beta'):
                continue
            object.__setattr__(self, name, _array(value, shape, name))


@dataclass(frozen=True)
class AdjustedBarra:
    source: BarraSnapshot
    risk: StructuredCovariance
    weights_source: str
    recovery_diagnostics: dict | None

    @property
    def predicted_beta(self):
        return self.risk.beta_after

    @property
    def market_weights(self):
        return self.risk.market_weights

    @property
    def diagnostics(self):
        return dict(self.risk.diagnostics)

    def for_symbols(self, symbols: Sequence[str]):
        symbols = _labels(symbols, 'symbols')
        locations = {symbol: i for i, symbol in enumerate(self.source.symbols)}
        missing = set(symbols) - locations.keys()
        if missing:
            raise ValueError(f'symbols absent from full risk model: {sorted(missing)}')
        return BarraView(self, symbols, tuple(locations[symbol] for symbol in symbols))

    def to_dense(self):
        return self.risk.to_dense()


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


def adjust_barra(snapshot: BarraSnapshot, *, lower_bound=.15, beta_consistency_tolerance=1e-8,
                 tolerance=1e-12, max_iterations=128) -> AdjustedBarra:
    """Use supplied w or recover it from beta; never clip or normalize weights.

    When both w and beta are supplied, require consistency with the input
    covariance. Missing w is the only trigger for recovery, never bad supplied w.
    """
    if not np.isfinite(beta_consistency_tolerance) or beta_consistency_tolerance <= 0:
        raise ValueError('beta_consistency_tolerance must be positive and finite')
    w, recovery = snapshot.market_weights, None
    if w is None:
        if snapshot.predicted_beta is None:
            raise ValueError('provide market_weights or predicted_beta')
        recovered = recover_from_factors(snapshot.factor_exposure, snapshot.factor_covariance,
                                        snapshot.specific_variance, snapshot.predicted_beta)
        recovery = dict(recovered.diagnostics)
        if not recovered.market_consistent:
            raise ValueError(f'recovered weights are not a long-only fully invested market: {recovery}')
        w = recovered.weights
    risk = adjust_from_factors(snapshot.factor_exposure, snapshot.factor_covariance,
                               snapshot.specific_variance, w, lower_bound=lower_bound,
                               tolerance=tolerance, max_iterations=max_iterations)
    if snapshot.predicted_beta is not None:
        error = float(np.max(np.abs(snapshot.predicted_beta - risk.beta_before)))
        scale = max(1., float(np.max(np.abs(snapshot.predicted_beta))))
        if error > beta_consistency_tolerance * scale:
            raise ValueError(f'supplied predicted_beta disagrees with X/F/d/w: max error={error:g}')
    return AdjustedBarra(snapshot, risk, 'recovered' if recovery is not None else 'provided', recovery)


class BarraView:
    """Exact principal risk for M selected names, with only O(MK+M) CVXPY data.

    Non-held names influence the market adjustment via h=U.T@w and v_out.
    Never normalize selected market weights or recalibrate on the subset.
    """

    def __init__(self, adjusted: AdjustedBarra, symbols, indices):
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
