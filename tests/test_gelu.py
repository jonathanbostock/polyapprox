import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.gelu import gelu, gelu_ev, gelu_poly_ev, gelu_prime, gelu_prime_ev


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_evs(mu, sigma):
    analytic_ev = gelu_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: gelu(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_prime_evs(mu, sigma):
    analytic_ev = gelu_prime_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: gelu_prime(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_poly_evs(n, mu, sigma):
    analytic_ev = gelu_poly_ev(n, mu, sigma)
    numerical_ev, err = quad(
        lambda x: x**n * gelu(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
