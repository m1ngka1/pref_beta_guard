"""Optional pandas adapter for Trade Planner-style FactorRiskData.

No dependency on trade_planner. The input object only needs the three named
factor_exposure/factor_covariance/specific_variance attributes.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import AdjustedBarra, BarraSnapshot, _labels, adjust_barra


def _day(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tz is not None:
        raise ValueError('dates must be valid timezone-naive model dates')
    return stamp.normalize()


def _normalize(value, kind):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            day = _day(key)
            if day in result:
                raise ValueError(f'{kind}: duplicate normalized date {day}')
            result[day] = _normalize(item, 'factor_covariance_static')
        return result
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value = value.copy()
        if isinstance(value.index, pd.MultiIndex):
            if value.index.nlevels != 2:
                raise ValueError(f'{kind}: expected a two-level (date, label) index')
            value.index = pd.MultiIndex.from_tuples([(_day(d), label) for d, label in value.index])
        elif kind not in ('factor_exposure', 'factor_covariance', 'factor_covariance_static') and isinstance(value, pd.DataFrame):
            value.index = pd.DatetimeIndex([_day(d) for d in value.index])
        if not value.index.is_unique or (isinstance(value, pd.DataFrame) and not value.columns.is_unique):
            raise ValueError(f'{kind}: duplicate row or column labels')
    return value


def _frame_at(value, day, name):
    if isinstance(value.index, pd.MultiIndex):
        try:
            return value.xs(day, level=0)
        except KeyError as exc:
            raise ValueError(f'{name}: missing date {day.date()}') from exc
    return value


def _covariance_at(value, day, t, count, factors):
    if isinstance(value, Mapping):
        if day not in value:
            raise ValueError(f'factor_covariance: missing date {day.date()}')
        value = value[day]
    if isinstance(value, pd.DataFrame):
        frame = _frame_at(value, day, 'factor_covariance')
        if set(frame.index) != set(factors) or set(frame.columns) != set(factors):
            raise ValueError('factor_covariance labels must match all exposure factors')
        return frame.reindex(index=factors, columns=factors).to_numpy(float)
    array = np.asarray(value, dtype=float)
    k = len(factors)
    if array.shape == (count, k, k):
        return array[t]
    if array.shape == (k, k):
        return array
    raise ValueError('factor_covariance must be KxK or TxKxK')


def _vector_at(value, day, t, count, symbols, name):
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        if isinstance(value.index, pd.MultiIndex):
            frame = _frame_at(value, day, name)
            if name not in frame.columns:
                raise ValueError(f'{name}: missing value column {name}')
            value = frame[name]
        else:
            if day not in value.index:
                raise ValueError(f'{name}: missing date {day.date()}')
            value = value.loc[day]
    if isinstance(value, pd.Series):
        value = _frame_at(value, day, name)
        return value.reindex(symbols).to_numpy(float)
    array = np.asarray(value, dtype=float)
    n = len(symbols)
    if array.shape == (n,):
        return array
    if array.shape == (count, n):
        return array[t]
    if array.shape == (n, 1):
        return array[:, 0]
    raise ValueError(f'{name} must have shape (N,), (N,1) or (T,N)')


@dataclass(frozen=True)
class AdjustedBarraPanel:
    snapshots: tuple[AdjustedBarra, ...]

    def __post_init__(self):
        items = tuple(self.snapshots)
        if not items or len({item.source.as_of for item in items}) != len(items):
            raise ValueError('panel snapshots must be nonempty with unique dates')
        first = items[0].source
        for item in items[1:]:
            if any(getattr(item.source, key) != getattr(first, key) for key in
                   ('model_id', 'currency', 'horizon', 'symbols', 'factor_names')):
                raise ValueError('panel snapshots must share provenance and ordered labels')
        object.__setattr__(self, 'snapshots', items)

    def at(self, date):
        key = _day(date).date().isoformat()
        for snapshot in self.snapshots:
            if snapshot.source.as_of == key:
                return snapshot
        raise ValueError(f'adjusted risk missing date {key}; no implicit forward fill')

    @property
    def dates(self):
        return tuple(item.source.as_of for item in self.snapshots)

    def beta_frame(self, symbols=None):
        names = self.snapshots[0].source.symbols if symbols is None else tuple(symbols)
        return pd.DataFrame([item.for_symbols(names).predicted_beta for item in self.snapshots],
                            index=pd.to_datetime(self.dates), columns=names)


def adjust_factor_risk_data(data, *, dates, model_id, currency, horizon,
                           market_weights=None, predicted_beta=None, lower_bound=.15,
                           beta_consistency_tolerance=1e-8) -> AdjustedBarraPanel:
    """Adjust full-universe FactorRiskData before selecting tradable symbols.

    Preserve first requested date's X row order; require the same universe on
    other dates. Labeled inputs are aligned; unlabelled arrays must already use
    that symbol order, X's factor order, and the supplied dates order.
    No annualization, filling, market normalization, or factor filtering.
    """
    days = tuple(_day(d) for d in dates)
    if not days or len(set(days)) != len(days):
        raise ValueError('dates must be nonempty and unique after normalization')
    X = _normalize(data.factor_exposure, 'factor_exposure')
    if not isinstance(X, pd.DataFrame):
        raise ValueError('factor_exposure must be a labeled DataFrame; use BarraSnapshot for arrays')
    F = _normalize(data.factor_covariance, 'factor_covariance')
    fields = {name: _normalize(value, name) for name, value in {
        'specific_variance': data.specific_variance,
        'market_weights': market_weights, 'predicted_beta': predicted_beta,
    }.items()}
    first = _frame_at(X, days[0], 'factor_exposure')
    symbols = _labels(first.index, 'symbols')
    factors = _labels(first.columns, 'factor_names')
    outputs = []
    for t, day in enumerate(days):
        frame = _frame_at(X, day, 'factor_exposure')
        if set(frame.index) != set(symbols):
            raise ValueError('full model universe changes across dates; prepare separate panels')
        snapshot = BarraSnapshot(
            as_of=day.date().isoformat(), model_id=model_id, currency=currency, horizon=horizon,
            symbols=symbols, factor_names=factors,
            factor_exposure=frame.reindex(index=symbols, columns=factors).to_numpy(float),
            factor_covariance=_covariance_at(F, day, t, len(days), factors),
            **{name: _vector_at(value, day, t, len(days), symbols, name) for name, value in fields.items()},
        )
        try:
            outputs.append(adjust_barra(snapshot, lower_bound=lower_bound,
                                         beta_consistency_tolerance=beta_consistency_tolerance))
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f'{model_id} {snapshot.as_of}: {exc}') from exc
    return AdjustedBarraPanel(tuple(outputs))
