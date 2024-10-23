from hypothesis import given, settings, strategies as st

from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

from polyapprox.gelu import gelu, gelu_prime, gelu_ev, gelu_prime_ev


@given(st.floats(-10, 10), st.floats(0.1, 10))
def test_gelu_evs(mu, sigma):
    analytic_ev = gelu_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: gelu(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    # print(f"{analytic_ev=}, {numerical_ev=}, {mu=}, {sigma=}, {err=}")
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)


@given(st.floats(-10, 10), st.floats(0.1, 10))
def test_gelu_prime_evs(mu, sigma):
    analytic_ev = gelu_prime_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: gelu_prime(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)