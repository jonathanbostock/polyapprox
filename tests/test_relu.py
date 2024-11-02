from hypothesis import given, strategies as st

from scipy.integrate import quad
from scipy.stats import norm
import numpy as np

from polyapprox.relu import relu, relu_prime, relu_ev, relu_prime_ev, relu_poly_ev


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_evs(mu, sigma):
    analytic_ev = relu_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: relu(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_prime_evs(mu, sigma):
    analytic_ev = relu_prime_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: relu_prime(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
