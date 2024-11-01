from hypothesis import given, strategies as st

from scipy.integrate import dblquad
from scipy.stats import multivariate_normal as mvn
import numpy as np

from polyapprox.integrate import bivariate_normal_cdf, isserlis


@given(st.floats(-0.999, 0.999), st.floats(-1e6, 1e6), st.floats(-1e6, 1e6))
def test_bivariate_normal_cdf(rho, x, y):
    ours = bivariate_normal_cdf(x, y, rho)
    scipy = mvn.cdf([x, y], cov=[[1, rho], [rho, 1]])

    # Sometimes SciPy returns NaNs for large values while we don't
    # don't penalize us for being more numerically stable
    if not np.isnan(scipy):
        np.testing.assert_allclose(ours, scipy, atol=1e-12)


def test_isserlis():
    B = np.random.randn(2, 2) / np.sqrt(2)
    K = B @ B.T

    analytical = isserlis(K, [0] * 4 + [1] * 6)
    numerical, error = dblquad(
        lambda y, x: x ** 4 * y ** 6 * mvn.pdf([x, y], cov=K), -np.inf, np.inf, -np.inf, np.inf,
    )
    np.testing.assert_allclose(numerical, analytical, atol=error)