import cvxpy as cp
import numpy as np
import pytest

from beta_guard import (
    InfeasibleAdjustmentError,
    SolverFailureError,
    adjust_factor_covariance,
)


def model():
    X = np.column_stack([np.ones(4), [-1.1, -0.9, 0.8, 1.2]])
    F = np.array([[1.0, 1.0], [1.0, 2.0]])
    D = np.full(4, 0.04)
    w = np.full(4, 0.25)
    return X, F, D, w


def check_invariants(X, F, D, w, result, floor=0.15):
    D = np.diag(D) if np.ndim(D) == 1 else D
    sigma = X @ F @ X.T + D
    q = X.T @ w
    s = w @ sigma @ w
    t = q @ F @ q
    np.testing.assert_allclose(result.covariance, X @ result.factor_covariance @ X.T + D)
    new_s = w @ result.covariance @ w
    actual = result.covariance @ w / new_s
    assert abs(new_s / s - 1) < 1e-9
    assert abs(q @ result.factor_covariance @ q / t - 1) < 1e-9
    assert abs(q @ result.h) < 1e-9
    assert actual.min() >= floor - 1e-7
    np.testing.assert_allclose(actual, result.beta_target, atol=1e-8)
    np.testing.assert_allclose(result.beta_after, actual, atol=1e-12)
    np.testing.assert_allclose(result.beta_target, result.beta_before + X @ result.h, atol=1e-12)
    F_new = result.factor_covariance
    scale = max(np.max(np.abs(F)), np.max(np.abs(F_new)))
    old_residual = F - np.outer(F @ q, F @ q) / t
    new_residual = F_new - np.outer(F_new @ q, F_new @ q) / t
    np.testing.assert_allclose(new_residual, old_residual, atol=scale * 1e-9)
    assert np.linalg.eigvalsh(F_new / scale).min() >= -1e-10
    assert np.linalg.eigvalsh(result.covariance / scale).min() >= -1e-10


def test_two_factor_case_matches_analytic_optimum():
    X, F, D, w = model()
    saved = [v.copy() for v in (X, F, D, w)]
    result = adjust_factor_covariance(X, F, D, w, check_psd=True)
    expected_h1 = (0.15 - result.beta_before[0]) / X[0, 1]
    np.testing.assert_allclose(result.h, [0, expected_h1], atol=1e-8)
    assert result.status == "optimal"
    check_invariants(X, F, D, w, result)
    for actual, original in zip((X, F, D, w), saved):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("preserve_order,pin_negative", [(True, False), (False, True), (True, True)])
def test_optional_constraints(preserve_order, pin_negative):
    X, F, D, w = model()
    result = adjust_factor_covariance(
        X, F, D, w, preserve_order=preserve_order, pin_negative=pin_negative,
        objective_weights=[1, 2, 3, 4], check_psd=True,
    )
    check_invariants(X, F, D, w, result)
    if preserve_order:
        assert np.diff(result.beta_after[np.argsort(result.beta_before)]).min() >= -1e-8
    if pin_negative:
        np.testing.assert_allclose(result.beta_after[result.beta_before < 0], 0.15, atol=1e-8)


def test_full_factor_space_matches_known_name_by_name_solution():
    w = np.full(4, 0.25)
    beta = np.array([-0.1, 0.1, 1.8, 2.2])
    residual = 0.04 * (np.eye(4) - np.ones((4, 4)) / 4)
    F = 0.09 * np.outer(beta, beta) + residual
    result = adjust_factor_covariance(
        np.eye(4), F, np.zeros(4), w, objective_weights=w,
        preserve_order=True, pin_negative=True, check_psd=True,
    )
    target = np.array([0.15, 0.15, 1.65, 2.05])
    np.testing.assert_allclose(result.beta_after, target, atol=1e-8)
    np.testing.assert_allclose(result.covariance, residual + 0.09 * np.outer(target, target), atol=1e-8)


def test_preserve_order_keeps_original_ties_despite_different_penalties():
    w = np.full(4, 0.25)
    beta = np.array([-0.1, 0.5, 0.5, 3.1])
    F = np.outer(beta, beta) + 0.04 * (np.eye(4) - np.ones((4, 4)) / 4)
    result = adjust_factor_covariance(
        np.eye(4), F, np.zeros(4), w, objective_weights=[1, 1, 10, 1],
        preserve_order=True,
    )
    np.testing.assert_allclose(result.beta_after, [0.15, 7 / 15, 7 / 15, 35 / 12], atol=1e-7)
    assert abs(result.beta_after[1] - result.beta_after[2]) < 1e-8


def test_one_factor_infeasible_problem_is_explicit():
    with pytest.raises(InfeasibleAdjustmentError):
        adjust_factor_covariance([[1], [-1]], [[1]], [0.01, 0.01], [0.75, 0.25])


def test_pinning_multiple_negative_betas_can_make_feasible_floor_infeasible():
    X = np.column_stack([np.ones(4), [-2, -1, 1, 2]])
    F = np.array([[1, 1.5], [1.5, 3]])
    D, w = np.full(4, 0.04), np.full(4, 0.25)
    result = adjust_factor_covariance(X, F, D, w, preserve_order=True)
    check_invariants(X, F, D, w, result)
    with pytest.raises(InfeasibleAdjustmentError):
        adjust_factor_covariance(X, F, D, w, pin_negative=True)


def test_correlated_specific_covariance_and_nonmarket_asset():
    X, F, _, _ = model()
    w = np.array([0, 0.2, 0.3, 0.5])
    D = 0.04 * np.eye(4) + 0.01 * np.ones((4, 4))
    result = adjust_factor_covariance(X, F, D, w, check_psd=True)
    check_invariants(X, F, D, w, result)


def test_rank_deficient_exposures():
    X, _, D, w = model()
    X = np.column_stack([X, X[:, 1]])
    F = np.array([[1, 0.6, 0.4], [0.6, 1, 0], [0.4, 0, 1]])
    result = adjust_factor_covariance(X, F, D, w, preserve_order=True, check_psd=True)
    check_invariants(X, F, D, w, result)


def test_singular_factor_covariance():
    X, _, D, w = model()
    F = np.ones((2, 2))
    result = adjust_factor_covariance(X, F, D, w, check_psd=True)
    check_invariants(X, F, D, w, result)


@pytest.mark.parametrize("scale", [1e-12, 1, 1e12])
def test_scale_invariance(scale):
    X, F, D, w = model()
    result = adjust_factor_covariance(X, F * scale, D * scale, w, check_psd=True)
    reference = adjust_factor_covariance(X, F, D, w)
    np.testing.assert_allclose(result.beta_after, reference.beta_after, atol=1e-8)
    check_invariants(X, F * scale, D * scale, w, result)


def test_no_change_does_not_call_solver(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not call solver")
    monkeypatch.setattr(cp.Problem, "solve", fail)
    result = adjust_factor_covariance(np.ones((3, 1)), [[0.04]], [0.01] * 3, [1 / 3] * 3)
    assert result.status == "unchanged"
    np.testing.assert_array_equal(result.factor_covariance, [[0.04]])


def test_unavailable_solver_is_explicit():
    with pytest.raises(SolverFailureError, match="failed"):
        adjust_factor_covariance(*model(), solver="NOT_A_SOLVER")


def test_inaccurate_status_is_not_returned_as_success(monkeypatch):
    def inaccurate(self, **kwargs):
        self._status = cp.OPTIMAL_INACCURATE
    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    with pytest.raises(SolverFailureError, match="optimal_inaccurate"):
        adjust_factor_covariance(*model())


def test_zero_market_factor_variance_rejected():
    with pytest.raises(ValueError, match="variance t"):
        adjust_factor_covariance([[-1], [1]], [[1]], [0.1, 0.1], [0.5, 0.5])


@pytest.mark.parametrize("weights", [[0, 1, 0, 0], [1, 2], [1, 1, 1, -1]])
def test_invalid_objective_weights(weights):
    with pytest.raises(ValueError, match="objective_weights"):
        adjust_factor_covariance(*model(), objective_weights=weights)


def test_negative_specific_variance_rejected():
    X, F, D, w = model()
    D[0] = -0.1
    with pytest.raises(ValueError, match="specific variances"):
        adjust_factor_covariance(X, F, D, w)


def test_indefinite_factor_covariance_rejected():
    X, _, D, w = model()
    with pytest.raises(ValueError, match="positive semidefinite"):
        adjust_factor_covariance(X, [[1, 2], [2, 1]], D, w)


def test_invalid_weights_rejected():
    X, F, D, _ = model()
    with pytest.raises(ValueError, match="market_weights"):
        adjust_factor_covariance(X, F, D, [0.25] * 3 + [0.2])
