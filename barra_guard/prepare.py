"""Process one latest, already-aligned Barra snapshot before context assembly.

No dates, loaders or planner dependencies. Reuse the returned risk object for
every planning day. Original X/F/d remain owned by the caller's data layer.
"""

from dataclasses import dataclass

import numpy as np

from market_recovery import recover_from_factors
from market_weights import DEFAULT_WEIGHT_TOLERANCE
from stock_covariance import StructuredCovariance, adjust_from_factors
from ._validation import _array, _labels


@dataclass(frozen=True)
class AdjustedBarra:
    risk: StructuredCovariance
    symbols: tuple[str, ...] | None
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

    def for_symbols(self, symbols):
        """Prepare once in the downstream holding order, then reuse each day."""
        from .risk import BarraView
        if self.symbols is None:
            raise ValueError('provide symbols when calling adjust_barra to select by name')
        symbols = _labels(symbols, 'symbols')
        locations = {symbol: i for i, symbol in enumerate(self.symbols)}
        missing = set(symbols) - locations.keys()
        if missing:
            raise ValueError(f'symbols absent from full risk model: {sorted(missing)}')
        return BarraView(self, symbols, tuple(locations[symbol] for symbol in symbols))

    def to_dense(self):
        return self.risk.to_dense()


def adjust_barra(factor_exposure, factor_covariance, specific_variance, *,
                 market_weights=None, predicted_beta=None, symbols=None,
                 lower_bound=.15, beta_consistency_tolerance=1e-8,
                 weight_tolerance=DEFAULT_WEIGHT_TOLERANCE,
                 tolerance=1e-12, max_iterations=128) -> AdjustedBarra:
    """X: NxK; F: KxK; d: N-vector or diagonal NxN matrix. No time axis.

    Supply w or original beta. With both, check model/beta consistency. Recover
    weights ONLY when w is absent. Both paths clean weight errors within
    weight_tolerance and reject larger errors. When beta is supplied, recheck
    it against the cleaned weights using beta_consistency_tolerance.
    Optional symbols labels rows; all inputs must already be aligned.
    """
    X = np.asarray(factor_exposure, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError('factor_exposure must be one finite NxK snapshot, with no time axis')
    n = len(X)
    if symbols is not None:
        symbols = _labels(symbols, 'symbols')
        if len(symbols) != n:
            raise ValueError('symbols must label every row of factor_exposure')
    if not np.isfinite(beta_consistency_tolerance) or beta_consistency_tolerance <= 0:
        raise ValueError('beta_consistency_tolerance must be positive and finite')
    beta = None if predicted_beta is None else _array(predicted_beta, (n,), 'predicted_beta')
    w, recovery = market_weights, None
    if w is None:
        if beta is None:
            raise ValueError('provide market_weights or predicted_beta')
        recovered = recover_from_factors(
            X, factor_covariance, specific_variance, beta,
            consistency_tolerance=weight_tolerance,
        )
        recovery = dict(recovered.diagnostics)
        if not recovered.market_consistent:
            raise ValueError(f'recovered weights are not a long-only fully invested market: {recovery}')
        w = recovered.weights
    risk = adjust_from_factors(X, factor_covariance, specific_variance, w,
                               lower_bound=lower_bound, tolerance=tolerance, max_iterations=max_iterations,
                               weight_tolerance=weight_tolerance)
    if beta is not None:
        error = float(np.max(np.abs(beta - risk.beta_before)))
        risk.diagnostics["input_beta_consistency_max_error"] = error
        if error > beta_consistency_tolerance * max(1., float(np.max(np.abs(beta)))):
            raise ValueError(f'supplied predicted_beta disagrees with X/F/d/w: max error={error:g}')
    return AdjustedBarra(risk, symbols, 'recovered' if recovery is not None else 'provided', recovery)
