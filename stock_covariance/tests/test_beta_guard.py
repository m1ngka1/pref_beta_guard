from itertools import combinations

import numpy as np
import pytest

from beta_guard import NumericalError, adjust_covariance, calibrate_betas


def covariance_with_beta(beta, weights, market_variance=0.09):
    beta, weights = np.asarray(beta, float), np.asarray(weights, float)
    residual = 0.04 * (np.eye(beta.size) - np.outer(weights, weights) / (weights @ weights))
    return market_variance * np.outer(beta, beta) + residual


def check_invariants(sigma, weights, result, lower_bound=0.15):
    w = np.asarray(weights)
    old_s = w @ sigma @ w
    new_s = w @ result.covariance @ w
    actual = result.covariance @ w / new_s
    np.testing.assert_allclose(actual, result.beta_target, atol=2e-10, rtol=2e-10)
    np.testing.assert_allclose(result.beta_after, actual, atol=1e-13)
    assert actual.min() >= lower_bound - 2e-10
    assert abs(new_s / old_s - 1) < 2e-10
    assert abs(w @ result.beta_target - 1) < 2e-12
    order = np.argsort(result.beta_before)
    assert np.min(np.diff(result.beta_target[order])) >= 0
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-14)
    scale = np.max(np.abs(result.covariance))
    assert np.linalg.eigvalsh(result.covariance / scale).min() >= -1e-10
    old_residual = sigma - old_s * np.outer(result.beta_before, result.beta_before)
    new_residual = result.covariance - new_s * np.outer(actual, actual)
    np.testing.assert_allclose(new_residual, old_residual, atol=scale * 2e-10, rtol=2e-10)


def test_known_four_stock_example():
    w = np.full(4, 0.25)
    sigma = covariance_with_beta([-0.1, 0.1, 1.8, 2.2], w)
    saved = sigma.copy()
    result = adjust_covariance(sigma, w, check_psd=True)
    np.testing.assert_allclose(result.beta_before, [-0.1, 0.1, 1.8, 2.2], atol=1e-14)
    np.testing.assert_allclose(result.beta_after, [0.15, 0.15, 1.65, 2.05], atol=2e-12)
    assert abs(result.shift - 0.15) < 2e-12
    check_invariants(sigma, w, result)
    np.testing.assert_array_equal(sigma, saved)


def test_already_feasible_is_exact_noop():
    w = np.full(3, 1 / 3)
    sigma = covariance_with_beta([0.5, 1, 1.5], w)
    result = adjust_covariance(sigma, w)
    assert result.shift == result.iterations == 0
    np.testing.assert_array_equal(result.covariance, sigma)


def test_zero_weight_assets_are_adjusted_without_entering_calibration():
    w = np.array([0, 0.25, 0.25, 0.25, 0.25])
    sigma = covariance_with_beta([-5, -0.1, 0.1, 1.8, 2.2], w)
    result = adjust_covariance(sigma, w, check_psd=True)
    np.testing.assert_allclose(result.beta_after, [0.15, 0.15, 0.15, 1.65, 2.05], atol=1e-11)
    check_invariants(sigma, w, result)


def test_only_nonmarket_asset_needs_floor_so_no_compensation():
    w = np.array([0, 0.5, 0.5])
    sigma = covariance_with_beta([-1, 0.5, 1.5], w)
    result = adjust_covariance(sigma, w)
    assert result.shift == 0
    np.testing.assert_allclose(result.beta_after, [0.15, 0.5, 1.5], atol=1e-13)


def test_rank_one_covariance_is_supported():
    w = np.full(4, 0.25)
    beta = np.array([-0.1, 0.1, 1.8, 2.2])
    sigma = 0.09 * np.outer(beta, beta)
    result = adjust_covariance(sigma, w, check_psd=True)
    check_invariants(sigma, w, result)
    assert np.linalg.matrix_rank(result.covariance, tol=1e-10) == 1


@pytest.mark.parametrize("scale", [1e-12, 1, 1e12])
def test_covariance_units_do_not_change_beta_adjustment(scale):
    w = np.full(4, 0.25)
    sigma = scale * covariance_with_beta([-0.1, 0.1, 1.8, 2.2], w)
    result = adjust_covariance(sigma, w, check_psd=True)
    np.testing.assert_allclose(result.beta_after, [0.15, 0.15, 1.65, 2.05], atol=1e-11)
    check_invariants(sigma, w, result)


@pytest.mark.parametrize("seed", range(20))
def test_random_covariances(seed):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(12, 6))
    sigma = factors @ factors.T + 0.1 * np.eye(12)
    w = rng.uniform(size=12)
    w[:2] = 0
    w /= w.sum()
    result = adjust_covariance(sigma, w, check_psd=True)
    check_invariants(sigma, w, result)


def active_set_qp_oracle(beta, w, floor):
    """Independent small-QP oracle: enumerate active constraints and solve KKT systems."""
    n = len(beta)
    H = np.diag(w)
    best_value, best_beta = np.inf, None
    for size in range(n + 1):
        for active in combinations(range(n), size):
            E = np.vstack([w, np.eye(n)[list(active)]])
            rhs = np.r_[1.0, np.full(size, floor)]
            kkt = np.block([[H, E.T], [E, np.zeros((size + 1, size + 1))]])
            solution = np.linalg.lstsq(kkt, np.r_[H @ beta, rhs], rcond=None)[0][:n]
            if np.max(np.abs(E @ solution - rhs)) > 1e-9 or solution.min() < floor - 1e-9:
                continue
            value = 0.5 * np.sum(w * (solution - beta) ** 2)
            if value < best_value:
                best_value, best_beta = value, solution
    assert best_beta is not None
    return best_beta


@pytest.mark.parametrize("seed", range(5))
def test_bisection_matches_independent_qp_oracle(seed):
    rng = np.random.default_rng(seed + 100)
    w = rng.uniform(0.1, 1, 6)
    w /= w.sum()
    beta = rng.normal(size=6) * 2
    beta += 1 - w @ beta
    result = calibrate_betas(beta, w)
    expected = active_set_qp_oracle(beta, w, 0.15)
    np.testing.assert_allclose(result.beta, expected, atol=1e-9)


@pytest.mark.parametrize("floor", [1e-6, 0.15, 0.99])
def test_supported_floor_values(floor):
    beta = np.array([-0.1, 0.1, 1.8, 2.2])
    result = calibrate_betas(beta, np.full(4, 0.25), lower_bound=floor)
    assert result.beta.min() >= floor
    assert abs(result.beta.mean() - 1) < 1e-12


def test_iteration_exhaustion_is_not_returned_as_success():
    with pytest.raises(NumericalError, match="bisection"):
        calibrate_betas([-0.1, 0.1, 1.8, 2.2], np.full(4, 0.25), max_iterations=1)


@pytest.mark.parametrize("floor", [0, -0.1, 1, 2, np.nan])
def test_invalid_floor(floor):
    with pytest.raises(ValueError):
        calibrate_betas([0.5, 1.5], [0.5, 0.5], lower_bound=floor)


@pytest.mark.parametrize("weights", [[0.5, 0.4], [-0.5, 1.5], [1], [np.nan, 1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        calibrate_betas([0.5, 1.5], weights)


def test_inconsistent_external_beta_rejected():
    with pytest.raises(ValueError, match="beta must equal 1"):
        calibrate_betas([0.5, 0.5], [0.5, 0.5])


@pytest.mark.parametrize("sigma", [np.zeros((2, 2)), [[1, 0.1], [0, 1]], [[1, 0], [0, -1]]])
def test_invalid_covariance(sigma):
    with pytest.raises(ValueError):
        adjust_covariance(sigma, [0.5, 0.5], check_psd=True)


def test_single_stock():
    result = adjust_covariance([[0.04]], [1], check_psd=True)
    np.testing.assert_array_equal(result.beta_after, [1])
    assert result.iterations == 0
