"""Weight roundoff handling agrees across recovery and both adjustments."""

import numpy as np
import pytest

from barra_guard import adjust_barra
from factor_covariance import adjust_factor_covariance
from market_recovery import recover_from_covariance, recover_from_factors
from market_weights import prepare_market_weights
from stock_covariance import adjust_covariance, adjust_from_factors


def model(w):
    X = np.eye(len(w))
    F = np.eye(len(w)) + .1
    d = np.full(len(w), .1)
    sigma = F + np.diag(d)
    beta = sigma @ w / (w @ sigma @ w)
    return X, F, d, sigma, beta


@pytest.mark.parametrize('raw', [
    np.array([-2e-10, .4, .6 + 2e-10]),
    np.array([0., .4, .6]) * (1 + 3e-9),
    np.array([-2e-10, .4, .6 + 3e-9]),
])
def test_all_adjustments_use_clean_weights_without_mutating_input(raw):
    original = raw.copy()
    clean = np.maximum(raw, 0); clean /= clean.sum()
    X, F, d, sigma, beta = model(raw)
    results = [
        adjust_covariance(sigma, raw),
        adjust_from_factors(X, F, d, raw),
        adjust_factor_covariance(X, F, d, raw),
    ]
    for result in results:
        np.testing.assert_allclose(result.market_weights, clean, atol=1e-16)
        assert np.all(result.market_weights >= 0)
        assert abs(result.market_weights.sum() - 1) < 1e-15
        assert result.diagnostics['weight_correction_l1'] > 0
        dense = result.to_dense() if hasattr(result, 'to_dense') else result.covariance
        variance = clean @ dense @ clean
        np.testing.assert_allclose(result.market_variance, clean @ sigma @ clean, rtol=1e-12)
        np.testing.assert_allclose(variance, result.market_variance, rtol=1e-9)
        np.testing.assert_allclose(dense @ clean / variance, result.beta_after, atol=1e-10)
        assert result.beta_after.min() >= .15 - 1e-9
    for supplied in [raw, None]:
        result = adjust_barra(X, F, d, market_weights=supplied, predicted_beta=beta)
        np.testing.assert_allclose(result.market_weights, clean, atol=1e-14)
        assert result.diagnostics['input_beta_consistency_max_error'] < 1e-8
        if supplied is None:
            assert result.recovery_diagnostics['beta_reconstruction_max_error'] < 1e-12
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize('seed', range(8))
def test_recovered_zero_weight_names_no_longer_fail_on_roundoff(seed):
    from test_market_recovery import make_model
    X, F, d, w, _, _, beta = make_model(seed)
    recovered = adjust_barra(X, F, d, predicted_beta=beta)
    provided = adjust_barra(X, F, d, market_weights=w, predicted_beta=beta)
    assert np.all(recovered.market_weights >= 0)
    np.testing.assert_allclose(recovered.to_dense(), provided.to_dense(), atol=1e-10)


@pytest.mark.parametrize('raw', [
    [-2e-8, .4, .6 + 2e-8],
    [0., .4, .6 + 2e-8],
    [-6e-9, -6e-9, 1 + 12e-9],  # Total negative mass, not a per-name limit.
])
def test_outside_tolerance_is_rejected_everywhere(raw):
    raw = np.array(raw)
    X, F, d, sigma, beta = model(raw)
    assert not recover_from_covariance(sigma, beta).market_consistent
    assert not recover_from_factors(X, F, d, beta).market_consistent
    calls = [
        lambda: adjust_covariance(sigma, raw),
        lambda: adjust_from_factors(X, F, d, raw),
        lambda: adjust_factor_covariance(X, F, d, raw),
        lambda: adjust_barra(X, F, d, market_weights=raw, predicted_beta=beta),
        lambda: adjust_barra(X, F, d, predicted_beta=beta),
    ]
    for call in calls:
        with pytest.raises(ValueError, match='market'):
            call()


@pytest.mark.parametrize('recover', [False, True])
def test_cleanup_must_still_reproduce_supplied_beta(recover):
    raw = np.array([-2e-10, .4, .6 + 2e-10])
    X, F, d = np.eye(3), np.diag([1e6, 1., 1.]), np.ones(3)
    sigma = F + np.diag(d)
    beta = sigma @ raw / (raw @ sigma @ raw)
    assert recover_from_covariance(sigma, beta).market_consistent
    with pytest.raises(ValueError, match='disagrees'):
        adjust_barra(X, F, d, market_weights=None if recover else raw, predicted_beta=beta)


def test_custom_tolerance_propagates_to_recovery_and_adjustment():
    raw = np.array([-2e-8, .4, .6 + 2e-8])
    X, F, d, _, beta = model(raw)
    clean = np.maximum(raw, 0); clean /= clean.sum()
    for supplied in [raw, None]:
        result = adjust_barra(X, F, d, market_weights=supplied, predicted_beta=beta,
                              weight_tolerance=1e-7, beta_consistency_tolerance=1e-6)
        np.testing.assert_allclose(result.market_weights, clean, atol=1e-14)
    for call in [
        lambda: adjust_covariance(F + np.diag(d), raw, weight_tolerance=1e-7),
        lambda: adjust_from_factors(X, F, d, raw, weight_tolerance=1e-7),
        lambda: adjust_factor_covariance(X, F, d, raw, weight_tolerance=1e-7),
    ]:
        np.testing.assert_allclose(call().market_weights, clean, atol=1e-14)


def test_exact_tolerance_boundary_and_owned_output():
    tol = 2.**-26  # Exactly representable at sum=1.
    for raw in [np.array([-tol, 1 + tol]), np.array([0., 1 + tol])]:
        clean, _ = prepare_market_weights(raw, 2, tolerance=tol)
        np.testing.assert_array_equal(clean, [0., 1.])
        assert not np.shares_memory(clean, raw)
        with pytest.raises(ValueError):
            prepare_market_weights(raw, 2, tolerance=tol / 2)


@pytest.mark.parametrize('tolerance', [0, -1, np.nan, np.inf, 1])
def test_invalid_weight_tolerance(tolerance):
    w = np.array([.3, .7])
    X, F, d, _, beta = model(w)
    for supplied in [w, None]:
        with pytest.raises(ValueError, match='tolerance'):
            adjust_barra(X, F, d, market_weights=supplied, predicted_beta=beta,
                         weight_tolerance=tolerance)
