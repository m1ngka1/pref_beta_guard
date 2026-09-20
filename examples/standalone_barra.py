"""Raw inputs -> numeric context -> risk. Requires only NumPy, no solver."""

import numpy as np

from barra_guard import adjust_barra, covariance_matrix, portfolio_variance, risk_arrays


adjusted = adjust_barra(
    symbols=('A', 'B', 'C', 'D'),
    factor_exposure=np.array([[-.4, .3], [.1, -.2], [1.8, .4], [2.3, .1]]),
    factor_covariance=np.diag([.0004, .0001]), specific_variance=np.full(4, .0002),
    market_weights=np.full(4, .25),
)
view = adjusted.for_symbols(['C', 'A', 'B'])
context = {
    'symbols': view.symbols,
    'predicted_beta': view.predicted_beta.copy(),
    'adjusted_risk': risk_arrays(view),
}
del view, adjusted  # Downstream needs only numeric information in context.

p = np.array([.2, .3, .5])  # Future portfolio weights, in context symbol order.
variance = portfolio_variance(context['adjusted_risk'], p)
Sigma_star = covariance_matrix(context['adjusted_risk'])  # Optional matrix export.
np.testing.assert_allclose(variance, p @ Sigma_star @ p, atol=1e-12)
print('symbols:', context['symbols'])
print('beta*:', context['predicted_beta'])
print('variance:', variance)
