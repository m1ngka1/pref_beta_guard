"""Bounded synthetic benchmark; run with BLAS threads limited to one.

Reports stored-array / canonical sparse-array bytes, NOT process peak RSS.
Dense uses psd_wrap to exclude an unnecessary N-by-N PSD certification cost.
The structured construction proves PSD; correctness tests separately compare it
to the legacy dense API and exercise CVXPY without psd_wrap on small matrices.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import cvxpy as cp
import numpy as np
import osqp
from scipy import sparse

from stock_covariance import adjust_from_factors


def data_for(n, k, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    X[:, 0] += .8
    R = rng.normal(size=(k, k))
    F = .04 * (R @ R.T / k + np.eye(k))
    d = rng.uniform(.03, .08, n)
    w = np.full(n, 1/n)
    alpha = rng.normal(0, .02, n)
    return X, F, d, w, alpha


def sparse_stats(matrix):
    return {'shape': list(matrix.shape), 'nnz': matrix.nnz,
            'bytes': matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes}


def solve_one(kind, model, dense, alpha):
    n = model.n_assets
    p = cp.Variable(n)
    constraints = [cp.sum(p) == 1, p >= 0, p <= 5/n,
                   model.beta_after @ p >= .4, model.beta_after @ p <= 1.2]
    if kind == 'adjusted_structured':
        block = model.cvxpy_risk(p)
        variance = block.variance
        constraints.extend(block.constraints)
    elif kind == 'adjusted_dense':
        variance = cp.quad_form(p, cp.psd_wrap(dense))
    else:
        # Unadjusted factor-model baseline: same portfolio constraints and alpha.
        # Its optimum need not match, because its covariance differs.
        y = cp.Variable(model.n_factors)
        constraints.append(y == model.factor_risk_loadings.T @ p)
        variance = cp.sum_squares(y) + cp.sum(cp.multiply(model.specific_variances, cp.square(p)))
    problem = cp.Problem(cp.Minimize(3 * variance - alpha @ p), constraints)
    start = perf_counter()
    data, _, _ = problem.get_problem_data(cp.OSQP)
    compile_seconds = perf_counter() - start
    start = perf_counter()
    problem.solve(solver='OSQP', eps_abs=1e-8, eps_rel=1e-8,
                  max_iter=30000, polishing=True, warm_start=False)
    solve_wall_seconds = perf_counter() - start
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f'{kind}: {problem.status}')
    holding = p.value
    violation = max(abs(holding.sum() - 1), max(0, -holding.min()),
                    max(0, holding.max() - 5/n), max(0, .4 - model.beta_after @ holding),
                    max(0, model.beta_after @ holding - 1.2))
    if violation > 1e-6:
        raise AssertionError(f'holding constraint violation {violation}')
    if kind == 'adjusted_structured':
        if (data['P'] - sparse.diags(data['P'].diagonal())).nnz:
            raise AssertionError('structured quadratic term became nondiagonal')
        if data['A'].nnz > n * model.n_factors + 6*n + 3*model.n_factors + 3:
            raise AssertionError('structured equality coefficients grew unexpectedly')
    result = {
        'status': problem.status, 'compile_seconds': compile_seconds,
        'solve_wall_seconds': solve_wall_seconds,
        'compile_plus_solve_seconds': compile_seconds + solve_wall_seconds,
        'solver_reported_seconds': problem.solver_stats.solve_time,
        'iterations': problem.solver_stats.num_iters,
        'objective': float(problem.value), 'holding_constraint_violation': float(violation),
        'P': sparse_stats(data['P']), 'A': sparse_stats(data['A']), 'F': sparse_stats(data['F']),
    }
    result['canonical_sparse_bytes'] = sum(result[key]['bytes'] for key in ['P', 'A', 'F'])
    return result, holding


def run_case(n, k, repeats):
    X, F, d, w, alpha = data_for(n, k, seed=n)
    start = perf_counter()
    model = adjust_from_factors(X, F, d, w)
    prep = perf_counter() - start
    start = perf_counter()
    dense = model.to_dense()
    export = perf_counter() - start
    kinds = ['original_factor', 'adjusted_dense', 'adjusted_structured']
    runs = {kind: [] for kind in kinds}
    max_weight_error = max_objective_error = 0.
    for repeat in range(repeats):
        solutions = {}
        # Rotate order to reduce systematic first-run effects; no warm-start reuse.
        for kind in kinds[repeat % 3:] + kinds[:repeat % 3]:
            record, holding = solve_one(kind, model, dense, alpha)
            runs[kind].append(record)
            solutions[kind] = holding
        a, b = solutions['adjusted_dense'], solutions['adjusted_structured']
        max_weight_error = max(max_weight_error, float(np.max(np.abs(a-b))))
        max_objective_error = max(max_objective_error,
                                  abs(3*model.variance(a) - alpha@a - (3*model.variance(b) - alpha@b)))
    if max_weight_error > 2e-5 or max_objective_error > 2e-7:
        raise AssertionError(f'dense/structured disagreement: {max_weight_error}, {max_objective_error}')
    medians = {}
    for kind, records in runs.items():
        medians[kind] = {key: float(np.median([r[key] for r in records])) for key in
                         ['compile_seconds', 'solve_wall_seconds', 'compile_plus_solve_seconds',
                          'solver_reported_seconds', 'iterations', 'canonical_sparse_bytes']}
    return {
        'n_assets': n, 'n_factors': k, 'repeats': repeats,
        'structured_prepare_seconds': prep, 'dense_export_seconds': export,
        'structured_stored_array_bytes': model.storage_bytes, 'dense_stored_array_bytes': dense.nbytes,
        'negative_beta_before': int(np.sum(model.beta_before < 0)),
        'beta_floor_after': float(model.beta_after.min()),
        'dense_structured_max_weight_error': max_weight_error,
        'dense_structured_max_objective_error': float(max_objective_error),
        'medians': medians, 'runs': runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 300, 1000])
    parser.add_argument('--factors', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('structured_benchmark_results.json'))
    args = parser.parse_args()
    if min(args.sizes) < 10 or args.factors < 1 or args.repeats < 1:
        parser.error('sizes >= 10, factors >= 1, repeats >= 1 required')
    result = {
        'python': platform.python_version(), 'numpy': np.__version__, 'cvxpy': cp.__version__,
        'osqp': osqp.__version__, 'machine': platform.machine(),
        'notes': ['Synthetic inputs only; cold CVXPY problems, rotated run order.',
                  'Array bytes exclude input retention, intermediate arrays, solver workspace and RSS.',
                  'Dense uses psd_wrap; factor baseline has a different covariance/optimum.',
                  'Performance depends on factor count, conditioning, solver and portfolio constraints.'],
        'cases': [],
    }
    for n in args.sizes:
        case = run_case(n, args.factors, args.repeats)
        result['cases'].append(case)
        print(json.dumps({'n': n, 'medians': case['medians'],
                          'max_weight_error': case['dense_structured_max_weight_error']}, indent=2), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
