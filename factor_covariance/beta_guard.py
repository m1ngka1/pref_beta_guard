"""CVXPY beta adjustment in factor space, followed by a closed-form F update."""

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class InfeasibleAdjustmentError(RuntimeError):
    """The requested beta constraints cannot be met with fixed X and D."""


class SolverFailureError(RuntimeError):
    """The QP did not return an optimal solution or failed numerical checks."""


@dataclass(frozen=True)
class AdjustmentResult:
    factor_covariance: FloatArray
    covariance: FloatArray
    beta_before: FloatArray
    beta_target: FloatArray
    beta_after: FloatArray
    h: FloatArray
    market_variance: float
    market_factor_variance: float
    status: str
    objective_value: float
    diagnostics: dict[str, float]


def _symmetric(value: ArrayLike, name: str, check_psd: bool) -> FloatArray:
    matrix = np.asarray(value, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all()):
        raise ValueError(f"{name} must be a nonempty, finite square matrix")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    matrix = 0.5 * matrix + 0.5 * matrix.T
    if check_psd and np.linalg.eigvalsh(matrix / scale)[0] < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def adjust_factor_covariance(
    loadings: ArrayLike,
    factor_covariance: ArrayLike,
    specific_covariance: ArrayLike,
    market_weights: ArrayLike,
    *,
    lower_bound: float = 0.15,
    objective_weights: ArrayLike | None = None,
    preserve_order: bool = False,
    pin_negative: bool = False,
    solver: str = "CLARABEL",
    solver_options: dict[str, Any] | None = None,
    feasibility_tolerance: float = 1e-7,
    check_psd: bool = False,
) -> AdjustmentResult:
    """Keep X, D and market variance fixed; solve for h, then update F.

    specific_covariance accepts either an N-vector of variances (NOT vols)
    or a full NxN PSD matrix. objective_weights is the positive diagonal of W.
    preserve_order adds monotonicity constraints and keeps exact beta ties tied.
    pin_negative fixes all originally negative betas at lower_bound.

    F is always PSD-checked (factor dimension). check_psd=True additionally
    checks dense D and the final stock covariance, which can cost O(N^3).
    Full D must be PSD even when this extra check is disabled.

    Infeasible QPs raise InfeasibleAdjustmentError. Solver errors, inaccurate
    statuses, or failed output checks raise SolverFailureError. No fallback,
    hidden eigenvalue repair, or relaxation of the requested constraints.
    """
    X = np.asarray(loadings, dtype=float)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("loadings must be a nonempty, finite NxK matrix")
    n, k = X.shape
    F = _symmetric(factor_covariance, "factor_covariance", True)
    if F.shape != (k, k):
        raise ValueError("factor_covariance shape does not match loadings")
    specific = np.asarray(specific_covariance, dtype=float)
    if specific.ndim == 1:
        if specific.shape != (n,) or not np.isfinite(specific).all() or np.any(specific < 0):
            raise ValueError("specific variances must be a finite nonnegative N-vector")
        D = np.diag(specific)
    else:
        D = _symmetric(specific, "specific_covariance", check_psd)
        if D.shape != (n, n):
            raise ValueError("specific_covariance shape does not match loadings")
    w = np.asarray(market_weights, dtype=float)
    if (w.shape != (n,) or not np.isfinite(w).all() or np.any(w < 0)
            or abs(float(w.sum()) - 1.0) > 1e-12):
        raise ValueError("market_weights must be a finite nonnegative N-vector summing to 1")
    if not np.isfinite(lower_bound) or not 0 < lower_bound < 1:
        raise ValueError("lower_bound must be strictly between 0 and 1")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be finite and positive")
    penalty = np.ones(n) if objective_weights is None else np.asarray(objective_weights, dtype=float)
    if penalty.shape != (n,) or not np.isfinite(penalty).all() or np.any(penalty <= 0):
        raise ValueError("objective_weights must be a finite strictly positive N-vector")

    q = X.T @ w
    c = F @ q
    t = float(q @ c)
    s = t + float(w @ D @ w)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("market common-factor variance t must be strictly positive")
    if not np.isfinite(s) or s <= 0:
        raise ValueError("market total variance s must be strictly positive")
    beta = (X @ c + D @ w) / s
    if not np.isfinite(beta).all() or abs(float(w @ beta) - 1.0) > 1e-10:
        raise ValueError("original beta could not be computed consistently")
    order = np.argsort(beta, kind="stable")
    negatives = beta < 0

    if np.min(beta) >= lower_bound:
        h_value = np.zeros(k)
        status = "unchanged"
    else:
        h = cp.Variable(k)
        delta = X @ h
        target = beta + delta
        q_unit = q / np.linalg.norm(q)
        constraints = [target >= lower_bound, q_unit @ h == 0]
        if preserve_order and n > 1:
            constraints.append(target[order[1:]] >= target[order[:-1]])
            tied = np.flatnonzero(np.diff(beta[order]) == 0)
            if tied.size:
                constraints.append(target[order[tied + 1]] == target[order[tied]])
        if pin_negative and np.any(negatives):
            constraints.append(target[negatives] == lower_bound)
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(cp.multiply(np.sqrt(penalty), delta))),
            constraints,
        )
        options = {}
        if solver == "CLARABEL":
            options.update(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
        options.update(solver_options or {})
        try:
            problem.solve(solver=solver, **options)
        except cp.error.SolverError as exc:
            raise SolverFailureError(f"{solver} failed: {exc}") from exc
        status = str(problem.status)
        if status == cp.INFEASIBLE:
            raise InfeasibleAdjustmentError("beta constraints are infeasible with fixed X and D")
        if status != cp.OPTIMAL or h.value is None:
            raise SolverFailureError(f"QP did not return an optimal solution: {status}")
        h_value = np.asarray(h.value, dtype=float).reshape(k)
        # Remove only the solver's floating-point market-neutrality residual.
        # All requested constraints are checked again AFTER this projection.
        h_value = h_value - q_unit * float(q_unit @ h_value)

    target_beta = beta + X @ h_value
    if not np.isfinite(target_beta).all():
        raise SolverFailureError("QP returned a nonfinite beta")
    c_new = c + s * h_value
    F_perp = F - np.outer(c, c) / t
    if status == "unchanged":
        F_new = F.copy()
    else:
        F_new = F_perp + np.outer(c_new, c_new) / t
        F_new = 0.5 * F_new + 0.5 * F_new.T
    try:
        F_new = _symmetric(F_new, "updated factor_covariance", True)
    except ValueError as exc:
        raise SolverFailureError("updated factor covariance failed validation") from exc
    sigma_new = X @ F_new @ X.T + D
    sigma_new = 0.5 * sigma_new + 0.5 * sigma_new.T
    if not np.isfinite(sigma_new).all():
        raise SolverFailureError("nonfinite updated stock covariance")
    new_variance = float(w @ sigma_new @ w)
    if not np.isfinite(new_variance) or new_variance <= 0:
        raise SolverFailureError("nonpositive updated market variance")
    realized = sigma_new @ w / new_variance
    order_violation = (
        max(0.0, float(-np.min(np.diff(realized[order]))))
        if preserve_order and n > 1 else 0.0
    )
    tied_positions = np.flatnonzero(np.diff(beta[order]) == 0) if preserve_order else np.array([], dtype=int)
    tie_error = (
        float(np.max(np.abs(realized[order[tied_positions + 1]] - realized[order[tied_positions]])))
        if tied_positions.size else 0.0
    )
    pin_error = (
        float(np.max(np.abs(realized[negatives] - lower_bound)))
        if pin_negative and np.any(negatives) else 0.0
    )
    diagnostics = {
        "weighted_mean_error": abs(float(w @ target_beta - 1.0)),
        "market_variance_relative_error": abs(new_variance / s - 1.0),
        "beta_reconstruction_max_error": float(np.max(np.abs(realized - target_beta))),
        "floor_violation": max(0.0, lower_bound - float(np.min(realized))),
        "order_violation": order_violation,
        "tie_error": tie_error,
        "negative_pin_error": pin_error,
    }
    if (not np.isfinite(realized).all()
            or any(value > feasibility_tolerance for value in diagnostics.values())):
        raise SolverFailureError(f"updated covariance failed constraints: {diagnostics}")
    if check_psd:
        try:
            _symmetric(sigma_new, "updated covariance", True)
        except ValueError as exc:
            raise SolverFailureError("updated stock covariance failed the PSD check") from exc
    return AdjustmentResult(
        factor_covariance=F_new, covariance=sigma_new, beta_before=beta,
        beta_target=target_beta, beta_after=realized, h=h_value,
        market_variance=s, market_factor_variance=t, status=status,
        objective_value=float(0.5 * np.sum(penalty * (X @ h_value) ** 2)),
        diagnostics=diagnostics,
    )
