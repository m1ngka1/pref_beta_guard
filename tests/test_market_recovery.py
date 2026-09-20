import numpy as np
import pytest

from market_recovery import recover_from_covariance, recover_from_factors


def make_model(seed=0, n=24, k=5):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=(n, k - 1))])
    C = rng.normal(size=(k, k))
    F = C @ C.T / 100
    d = rng.uniform(0.01, 0.05, n)
    w = rng.uniform(size=n)
    w[:3] = 0
    w /= w.sum()
    sigma = X @ F @ X.T + np.diag(d)
    s = w @ sigma @ w
    beta = sigma @ w / s
    return X, F, d, w, sigma, s, beta


@pytest.mark.parametrize("seed", range(20))
def test_recover_known_weights_and_variance(seed):
    X, F, d, w, sigma, s, beta = make_model(seed)
    for result in (recover_from_covariance(sigma, beta), recover_from_factors(X, F, d, beta)):
        np.testing.assert_allclose(result.weights, w, atol=2e-12, rtol=2e-10)
        np.testing.assert_allclose(result.market_variance, s, rtol=2e-12)
        np.testing.assert_allclose(result.market_volatility, np.sqrt(s), rtol=2e-12)
        np.testing.assert_allclose(result.beta_recomputed, beta, atol=2e-11)
        assert result.market_consistent
        assert result.diagnostics['negative_weight_mass'] < 1e-12


def test_analytic_diagonal_case():
    sigma = np.diag([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    beta = np.array([0.2, 0.6, 1.5]) / 0.97
    result = recover_from_covariance(sigma, beta)
    np.testing.assert_allclose(result.weights, w, atol=1e-15)
    assert abs(result.market_variance - 0.97) < 1e-15


def test_singular_factor_covariance_is_allowed_with_positive_specific_variance():
    X, _, d, w, _, _, _ = make_model()
    F = np.ones((X.shape[1], X.shape[1])) * 0.01
    sigma = X @ F @ X.T + np.diag(d)
    s = w @ sigma @ w
    result = recover_from_factors(X, F, d, sigma @ w / s)
    np.testing.assert_allclose(result.weights, w, atol=1e-12)
    assert result.market_consistent


def test_zero_factor_covariance():
    X, F, d, w, _, _, _ = make_model()
    F = np.zeros_like(F)
    s = w @ (d * w)
    result = recover_from_factors(X, F, d, d * w / s)
    np.testing.assert_allclose(result.weights, w, atol=1e-14)


def test_diagonal_matrix_and_column_beta_inputs():
    X, F, d, w, sigma, s, beta = make_model()
    beta_column = beta[:, None]
    result = recover_from_factors(X, F, np.diag(d), beta_column)
    dense = recover_from_covariance(sigma, beta_column)
    assert result.weights.shape == w.shape
    np.testing.assert_allclose(result.weights, w, atol=1e-12)
    np.testing.assert_allclose(result.weights, dense.weights, atol=1e-12)
    np.testing.assert_allclose(result.market_variance, s, rtol=1e-12)


def test_full_correlated_specific_covariance_via_dense_path():
    X, F, d, w, _, _, _ = make_model()
    D = np.diag(d) + 0.003 * np.ones((len(w), len(w)))
    sigma = X @ F @ X.T + D
    s = w @ sigma @ w
    result = recover_from_covariance(sigma, sigma @ w / s)
    np.testing.assert_allclose(result.weights, w, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-12, 1e12])
def test_covariance_unit_scaling(scale):
    X, F, d, w, sigma, s, beta = make_model()
    for result in (recover_from_covariance(scale * sigma, beta),
                   recover_from_factors(X, scale * F, scale * d, beta)):
        np.testing.assert_allclose(result.weights, w, atol=1e-12)
        np.testing.assert_allclose(result.market_variance, scale * s, rtol=1e-11)


def test_scaled_beta_is_not_silently_normalized():
    _, _, _, w, sigma, s, beta = make_model()
    result = recover_from_covariance(sigma, 0.9 * beta)
    np.testing.assert_allclose(result.weights, w / 0.9, atol=1e-12)
    np.testing.assert_allclose(result.market_variance, s / 0.9**2, rtol=1e-12)
    assert not result.full_investment_consistent
    assert result.long_only_consistent


def test_stock_subset_does_not_recover_full_market():
    sigma = np.diag([1.0, 2.0, 3.0])
    w = np.array([0.2, 0.3, 0.5])
    beta = sigma @ w / (w @ sigma @ w)
    result = recover_from_covariance(sigma[:2, :2], beta[:2])
    assert not result.full_investment_consistent
    assert abs(result.weights.sum() - 1) > 0.1


def test_long_short_market_is_reported_not_clipped():
    sigma = np.diag([1.0, 2.0, 3.0])
    w = np.array([-0.1, 0.4, 0.7])
    beta = sigma @ w / (w @ sigma @ w)
    result = recover_from_covariance(sigma, beta)
    np.testing.assert_allclose(result.weights, w, atol=1e-15)
    assert result.full_investment_consistent
    assert not result.long_only_consistent
    assert result.weights[0] < 0


def test_rounded_beta_reports_budget_inconsistency():
    _, _, _, _, sigma, _, beta = make_model(9)
    result = recover_from_covariance(sigma, beta.round(3))
    assert not result.market_consistent
    # A good reconstruction alone is NOT evidence that the market was identified.
    np.testing.assert_allclose(result.beta_recomputed, beta.round(3), atol=1e-12)


def test_factor_path_never_solves_an_n_by_n_system(monkeypatch):
    X, F, d, w, _, _, beta = make_model(n=300, k=6)
    original_solve = np.linalg.solve
    def factor_only(matrix, rhs):
        assert matrix.shape == (6, 6)
        return original_solve(matrix, rhs)
    monkeypatch.setattr(np.linalg, 'solve', factor_only)
    result = recover_from_factors(X, F, d, beta)
    np.testing.assert_allclose(result.weights, w, atol=1e-11)


def test_recovered_positive_weights_work_in_existing_stock_adjustment():
    from stock_covariance.beta_guard import adjust_covariance
    w = np.full(4, 0.25)
    beta = np.array([-0.1, 0.1, 1.8, 2.2])
    sigma = 0.09 * np.outer(beta, beta) + 0.04 * (np.eye(4) - np.ones((4, 4)) / 4)
    recovered = recover_from_covariance(sigma, beta)
    adjusted = adjust_covariance(sigma, recovered.weights, check_psd=True)
    np.testing.assert_allclose(adjusted.beta_after, [0.15, 0.15, 1.65, 2.05], atol=1e-10)


@pytest.mark.parametrize("sigma", [np.ones((2, 2)), [[1, 0], [0, -1]], [[1, 1], [0, 1]]])
def test_non_spd_or_nonsymmetric_covariance_is_rejected(sigma):
    with pytest.raises(ValueError):
        recover_from_covariance(sigma, [0.5, 1.5])


@pytest.mark.parametrize("beta", [[0, 0], [np.nan, 1], [1]])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        recover_from_covariance(np.eye(2), beta)


@pytest.mark.parametrize("d", [[0.1, 0], [0.1, -0.1], [[0.1, 0.01], [0, 0.1]]])
def test_fast_path_requires_positive_diagonal_specific_covariance(d):
    with pytest.raises(ValueError, match='specific_covariance'):
        recover_from_factors(np.eye(2), np.eye(2), d, [0.5, 1.5])


def test_indefinite_factor_covariance_rejected():
    with pytest.raises(ValueError, match='positive semidefinite'):
        recover_from_factors(np.eye(2), [[1, 2], [2, 1]], [0.1, 0.1], [0.5, 1.5])
