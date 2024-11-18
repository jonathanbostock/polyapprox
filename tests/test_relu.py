import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.relu import relu, relu_ev, relu_poly_ev, relu_prime, relu_prime_ev


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
        lambda x: relu_prime(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_poly_evs(n, mu, sigma):
    analytic_ev = relu_poly_ev(n, mu, sigma)
    numerical_ev, err = quad(
        lambda x: x**n * relu(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    # Don't use relative tolerance because the expected value can be very small.
    # The sharpness of ReLU also makes quad have a hard time estimating the error.
    assert abs(numerical_ev - analytic_ev) < np.finfo(float).eps + err
