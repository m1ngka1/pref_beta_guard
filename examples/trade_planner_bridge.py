"""Read-only smoke test against an existing Trade Planner checkout.

No production source is patched. The dense adapter exercises TradePlanner.solve
unchanged. The structured branch assembles the same small test problem using
its real context/state/constraint/cost/scaling helpers plus risk blocks. That
branch is a verification harness, not a replacement production solve() method.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np
import pandas as pd

from barra_guard.planner import DensePlannerRiskAdapter, PlannerRiskAdapter
from barra_guard.tabular import adjust_factor_risk_data


def run(reference_root):
    sys.path.insert(0, str(reference_root))
    from trade_planner.config import TradePlannerConfig
    from trade_planner.costs import CompositeCostModel, QuadraticParticipationImpact
    from trade_planner.data import PlannerDataProvider, build_context_from_provider
    from trade_planner.participation import ParticipationCapModel
    from trade_planner.planner import TradePlanner

    rng = np.random.default_rng(2)
    names, factors = [f'S{i}' for i in range(30)], ['market', 'size', 'value', 'industry']
    X = rng.normal(size=(30, 4)); X[:, 0] += .8
    X = pd.DataFrame(X, index=names, columns=factors)
    F = pd.DataFrame(np.diag([.0004, .0002, .0001, .0001]), index=factors, columns=factors)
    d = pd.Series(.0003, index=names)
    w = pd.Series(1/30, index=names)

    class Provider(PlannerDataProvider):
        def load_price(self, symbols, dates):
            return pd.Series({name: 30. + i for i, name in enumerate(names)}).reindex(symbols)

        def load_adv_shares(self, symbols, dates):
            return pd.Series(20000., index=symbols)

        def load_factor_exposure(self, symbols, dates):
            return X.reindex(symbols)

        def load_factor_covariance(self, factor_names, dates):
            return F.reindex(index=factor_names, columns=factor_names)

        def load_specific_variance(self, symbols, dates):
            return d.reindex(symbols)

    provider = Provider()
    orders = pd.DataFrame({'target_shares': [2400., -1800., 1200.]}, index=['S9', 'S2', 'S0'])
    ctx = build_context_from_provider(orders, '2026-06-19', '2026-06-23', provider)
    # Separate full-model query: ctx.factor_exposure contains only order names.
    full_data = provider.load_factor_risk_data(names, ctx.dates)
    panel = adjust_factor_risk_data(full_data, dates=ctx.dates, market_weights=w,
                                    model_id='SYNTHETIC', currency='JPY', horizon='daily')
    results = []
    for scaling in ['none', 'per_name']:
        config = TradePlannerConfig(
            participation_model=ParticipationCapModel(), risk_model=DensePlannerRiskAdapter(panel),
            cost_model=CompositeCostModel((QuadraticParticipationImpact(),)),
            residual_risk_weight=.4, inventory_risk_weight=.7,
            numerical_scaling=scaling, solver='CLARABEL', verify_hard_constraints=True,
        )
        dense_result = TradePlanner(config).solve(ctx)
        dense_trades = dense_result.schedule.pivot(index='date', columns='symbol', values='trade_shares').reindex(
            index=ctx.dates, columns=ctx.symbols).to_numpy()

        adapter = PlannerRiskAdapter(panel)
        blocks = []  # Local to this one problem; never shared across solves.

        class Collector:
            def objective(self, shares, objective_ctx, t):
                block = adapter.risk_block(shares, objective_ctx, t)
                blocks.append(block)
                return block.variance

        planner = TradePlanner(replace(config, risk_model=Collector()))
        target = ctx.orders['target_shares'].reindex(ctx.symbols).to_numpy(float)
        caps = config.participation_model.caps(ctx)
        decision, state = planner._new_decision_state(target=target, caps=caps, ctx=ctx)
        for plugin in config.constraints:
            validate = getattr(plugin, 'validate', None)
            if callable(validate):
                updated = validate(ctx, state)
                if updated is not None:
                    target = np.asarray(updated, float)
                    decision, state = planner._new_decision_state(target=target, caps=caps, ctx=ctx)
        constraints = [item for plugin in config.constraints for item in plugin.constraints(ctx, state)]
        objective_ctx = planner._objective_context(ctx, state.share_scale)
        terms = planner._objective_terms(objective_ctx, state)
        objective = sum(terms)
        constraints.extend(item for block in blocks for item in block.constraints)
        # Same reference as TradePlanner._objective_multiplier, with auxiliaries
        # initialized before reading the expression value.
        capacity = caps.sum(axis=0)
        reference = np.divide(caps, capacity[None, :], out=np.zeros_like(caps), where=capacity[None, :] > 0) * target
        decision.value = reference / state.share_scale
        assert objective.value is None
        for block in blocks:
            block.set_reference_values()
        assert objective.value is not None
        multiplier = planner._objective_multiplier(total_objective=objective, decision_variable=decision,
                                                   share_scale=state.share_scale, target=target, caps=caps)
        problem = cp.Problem(cp.Minimize(multiplier * objective), constraints)
        planner._solve_problem(problem)
        trades = state.trades.value
        certificate = planner._hard_constraint_certificate(trades=trades, target=target, caps=caps)
        assert not planner._certificate_violations(certificate), certificate
        np.testing.assert_allclose(trades, dense_trades, atol=.002, rtol=2e-6)
        np.testing.assert_allclose(objective.value, dense_result.diagnostics['objective'], rtol=2e-7)
        cumulative = np.cumsum(trades, axis=0)
        numeric_risk = sum(.7 * adapter.variance(cumulative[t], ctx, t)
                           + .4 * adapter.variance(target-cumulative[t], ctx, t) for t in range(len(ctx.dates)))
        cvx_risk = sum((.7 if i % 2 == 0 else .4) * block.variance.value for i, block in enumerate(blocks))
        np.testing.assert_allclose(numeric_risk, cvx_risk, rtol=1e-9)
        qp, _, _ = problem.get_problem_data(cp.OSQP)
        results.append({
            'scaling': scaling, 'dense_status': dense_result.diagnostics['status'], 'structured_status': problem.status,
            'full_assets': 30, 'traded_assets': 3, 'dates': len(ctx.dates),
            'max_trade_difference_shares': float(np.max(np.abs(trades-dense_trades))),
            'relative_objective_difference': float(abs(objective.value/dense_result.diagnostics['objective']-1)),
            'objective_multiplier': multiplier, 'risk_blocks': len(blocks),
            'qp_variables': qp['P'].shape[0], 'qp_equality_nnz': qp['A'].nnz, **certificate,
        })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade-planner-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = run(args.trade_planner_root)
    output = json.dumps(results, indent=2) + '\n'
    print(output)
    if args.output:
        args.output.write_text(output)
