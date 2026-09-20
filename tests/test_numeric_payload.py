"""Context payloads remain portable numeric data, independent of any solver."""

import io
import subprocess
import sys

import numpy as np
import pytest

from barra_guard import adjust_barra, covariance_matrix, portfolio_variance, risk_arrays


@pytest.mark.parametrize('subset', [False, True])
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('scale', [1e-10, 1., 1e10])
def test_export_roundtrip_matches_adjusted_covariance(subset, seed, scale):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(12, 3)); X[:, 0] += .7
    F = scale * np.diag([.04, .03, .01])
    adjusted = adjust_barra(X, F, scale*np.full(12, .02), market_weights=np.full(12, 1/12),
                            symbols=[f'S{i}' for i in range(12)])
    risk = adjusted.for_symbols(['S5', 'S1', 'S0']) if subset else adjusted.risk
    expected = risk.to_dense()
    data = risk_arrays(risk)
    assert set(data) == {'U', 'd', 'w', 'delta', 'h', 'v_out'}
    assert all(isinstance(value, (np.ndarray, float)) for value in data.values())
    assert all(not value.flags.writeable for value in data.values() if isinstance(value, np.ndarray))
    assert not np.shares_memory(data['delta'], risk.delta)
    if not subset:
        assert data['v_out'] == 0
    del risk, adjusted
    # A context can persist just these numeric arrays; no class instance required.
    buffer = io.BytesIO()
    np.savez(buffer, **data)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as loaded:
        data = {key: loaded[key].copy() for key in loaded.files}
    data['v_out'] = float(data['v_out'])
    p = rng.normal(size=len(data['d']))
    np.testing.assert_allclose(covariance_matrix(data)/scale, expected/scale, atol=1e-10)
    np.testing.assert_allclose(portfolio_variance(data, p[:, None])/scale, p @ expected @ p/scale, atol=1e-10)
    with pytest.raises(ValueError):
        portfolio_variance(data, np.full(len(p), np.nan))


def test_preparation_and_risk_never_import_cvxpy():
    source = '''
import sys
class BlockCvxpy:
    def find_spec(self, fullname, *args):
        if fullname == 'cvxpy' or fullname.startswith('cvxpy.'):
            raise AssertionError('unexpected CVXPY import')
sys.meta_path.insert(0, BlockCvxpy())
import numpy as np
from barra_guard import adjust_barra, risk_arrays, portfolio_variance, covariance_matrix
a = adjust_barra([[-1.], [2.]], [[.04]], [.01,.01], market_weights=[.5,.5], symbols=['A','B'])
for risk in [a.risk, a.for_symbols(['B','A'])]:
    data = risk_arrays(risk)
    p = np.array([.4,.6])
    np.testing.assert_allclose(portfolio_variance(data,p), p @ covariance_matrix(data) @ p)
assert 'cvxpy' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', source], check=True, capture_output=True, text=True)
