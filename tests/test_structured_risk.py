import numpy as np
import pytest

from stock_covariance.beta_guard import adjust_covariance
from stock_covariance.structured import adjust_from_factors


def inputs(seed=0, n=25, k=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    X[:, 0] += 0.7
    R = rng.normal(size=(k, k))
    F = R @ R.T / k
    d = rng.uniform(0.05, 0.2, n)
    w = rng.uniform(size=n)
    w[:2] = 0
    w /= w.sum()
    return X, F, d, w


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('scale', [1e-10, 1.0, 1e10])
def test_dense_equivalence_and_invariants(seed, scale):
    X, F, d, w = inputs(seed)
    F, d = scale * F, scale * d
    sigma = X @ F @ X.T + np.diag(d)
    expected = adjust_covariance(sigma, w, check_psd=True)
    model = adjust_from_factors(X, F, d, w)
    dense = model.to_dense()
    np.testing.assert_allclose(dense / scale, expected.covariance / scale, atol=2e-10)
    np.testing.assert_allclose(model.beta_after, expected.beta_after, atol=1e-10)
    np.testing.assert_allclose(dense @ w / (w @ dense @ w), model.beta_after, atol=1e-10)
    assert model.beta_after.min() >= 0.15 - 1e-9
    assert abs(model.updated_market_variance / model.market_variance - 1) < 1e-9
    rng = np.random.default_rng(seed + 100)
    p = rng.normal(size=w.size)  # Neither long-only nor normalized; also covers active/dollar exposures.
    np.testing.assert_allclose(model.matvec(p) / scale, dense @ p / scale, atol=1e-9)
    np.testing.assert_allclose(model.variance(p) / scale, p @ dense @ p / scale, atol=1e-9)
    np.testing.assert_allclose(model.gradient(p) / scale, 2 * dense @ p / scale, atol=2e-9)
    np.testing.assert_allclose(model.diagonal() / scale, np.diag(dense) / scale, atol=1e-10)
    idx = np.array([8, 3, 14])
    np.testing.assert_allclose(model.to_dense(idx) / scale, dense[np.ix_(idx, idx)] / scale, atol=1e-10)
    assert model.storage_bytes == 8 * (X.size + 6 * w.size)


def test_gradient_finite_difference():
    model = adjust_from_factors(*inputs())
    p = np.linspace(-0.2, 0.3, model.n_assets)
    h = 1e-6
    numerical = np.array([
        (model.variance(p + h * e) - model.variance(p - h * e)) / (2 * h)
        for e in np.eye(model.n_assets)
    ])
    np.testing.assert_allclose(model.gradient(p), numerical, atol=2e-8)


def test_singular_model_and_column_inputs():
    X = np.array([[-1., 0], [2, 1], [3, 0]])
    F = np.diag([0.09, 0])
    d, w = np.zeros(3), np.full(3, 1 / 3)
    model = adjust_from_factors(X, F, np.diag(d), w[:, None])
    expected = adjust_covariance(X @ F @ X.T, w)
    np.testing.assert_allclose(model.to_dense(), expected.covariance, atol=1e-11)
    assert np.linalg.matrix_rank(model.to_dense(), tol=1e-10) == 1
    np.testing.assert_allclose(model.matvec(w[:, None]), model.matvec(w))


def test_owned_readonly_snapshots():
    X, F, d, w = inputs()
    model = adjust_from_factors(X, F, d[:, None], w)
    expected = model.to_dense()
    for array in (X, F, d, w):
        array[:] = 0
    np.testing.assert_array_equal(model.to_dense(), expected)
    with pytest.raises(ValueError):
        model.delta[0] = 99


def test_no_adjustment_and_zero_factor_risk():
    model = adjust_from_factors(np.ones((5, 2)), np.zeros((2, 2)), np.ones(5), np.full(5, .2))
    np.testing.assert_array_equal(model.delta, 0)
    np.testing.assert_allclose(model.to_dense(), np.eye(5))


def test_preparation_never_builds_asset_outer_product(monkeypatch):
    args = inputs(n=200, k=5)
    def forbidden(*args, **kwargs):
        raise AssertionError('unexpected dense asset operation')
    monkeypatch.setattr(np, 'outer', forbidden)
    monkeypatch.setattr(np, 'diag', forbidden)
    model = adjust_from_factors(*args)
    assert model.storage_bytes < 8 * 200 * 200
    assert np.isfinite(model.variance(np.ones(200)))


@pytest.mark.parametrize('indices', [[-1], [25], [2, 2], [1.5], [], [[1]]])
def test_invalid_selection(indices):
    with pytest.raises(ValueError):
        adjust_from_factors(*inputs()).to_dense(indices)


@pytest.mark.parametrize('case', ['indefinite', 'correlated', 'negative_d', 'negative_w', 'budget', 'nonfinite'])
def test_invalid_model(case):
    X, F, d, w = inputs()
    if case == 'indefinite':
        F = -np.eye(F.shape[0])
    elif case == 'correlated':
        d = np.diag(d)
        d[0, 1] = 1e-5
    elif case == 'negative_d':
        d[0] = -0.1
    elif case == 'negative_w':
        w[0], w[1] = -1e-15, 1e-15
    elif case == 'budget':
        w *= .9
    elif case == 'nonfinite':
        X[0, 0] = np.nan
    with pytest.raises(ValueError):
        adjust_from_factors(X, F, d, w)
