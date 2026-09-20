"""Optional duck-typed bridges; does not import or modify trade_planner.

Use risk_block() with a planner collecting both expression and constraints.
DensePlannerRiskAdapter is the explicit small-universe compatibility fallback
for the existing objective-only protocol.
"""

import numpy as np


class PlannerRiskAdapter:
    def __init__(self, panel):
        self.panel = panel
        self._views = {}

    def view_for_date(self, ctx, date_index):
        adjusted = self.panel.at(ctx.dates[date_index])
        key = (adjusted.source.as_of, tuple(ctx.symbols))
        if key not in self._views:
            self._views[key] = adjusted.for_symbols(ctx.symbols)
        return self._views[key]

    def risk_block(self, position_shares, ctx, date_index):
        import cvxpy as cp
        # ctx may already contain price * share_scale, as in Trade Planner's
        # _objective_context. Use exactly that price once; never rescale again.
        dollars = cp.multiply(ctx.price[date_index], position_shares)
        return self.view_for_date(ctx, date_index).cvxpy_risk(dollars, name=f'barra_{date_index}')

    def objective(self, position_shares, ctx, date_index):
        raise TypeError('Structured risk requires risk_block() AND its constraints; '
                        'use DensePlannerRiskAdapter explicitly for an objective-only planner')

    def covariance_for_date(self, ctx, date_index):
        return self.view_for_date(ctx, date_index).to_dense()

    def variance(self, position_shares, ctx, date_index):
        return self.view_for_date(ctx, date_index).variance(
            np.asarray(ctx.price[date_index]) * np.asarray(position_shares))


class DensePlannerRiskAdapter(PlannerRiskAdapter):
    """Explicit compatibility path: creates an MxM matrix for planner names.

    Does not impose an automatic size threshold or change solver settings.
    Risk-data changes require a new adapter. No hidden mutable constraint list.
    """

    def __init__(self, panel):
        super().__init__(panel)
        self._covariances = {}

    def covariance_for_date(self, ctx, date_index):
        view = self.view_for_date(ctx, date_index)
        key = (view.adjusted.source.as_of, view.symbols)
        if key not in self._covariances:
            self._covariances[key] = view.to_dense()
        return self._covariances[key].copy()

    def objective(self, position_shares, ctx, date_index):
        import cvxpy as cp
        dollars = cp.multiply(ctx.price[date_index], position_shares)
        return cp.quad_form(dollars, self.covariance_for_date(ctx, date_index))
