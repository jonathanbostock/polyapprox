from functools import partial

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.jump_relu import jump_relu_ev, jump_relu_poly_ev, jump_relu_prime_ev


def jump_relu(x):
    """JumpReLU(x) activation function, with theta = 1.0"""
    return np.where(x > 1.0, x, 0)


def jump_relu_prime(x):
    """Derivative of JumpReLU(x) with theta = 1.0"""
    return np.where(x > 1.0, 1.0, 0.0)


def torch_test(f, target, *args, atol=1e-12):
    res = f(*map(lambda x: torch.tensor(x, dtype=torch.double), args))
    assert abs(res - target) < atol + torch.finfo(torch.double).eps


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_jump_relu_evs(mu, sigma):
    analytic_ev = jump_relu_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: jump_relu(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
    torch_test(jump_relu_ev, numerical_ev, mu, sigma, atol=err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_prime_evs(mu, sigma):
    analytic_ev = jump_relu_prime_ev(mu, sigma)
    numerical_ev, err = quad(
        lambda x: jump_relu_prime(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err)
    torch_test(jump_relu_prime_ev, numerical_ev, mu, sigma, atol=err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_jump_relu_poly_evs(n, mu, sigma):
    analytic_ev = jump_relu_poly_ev(n, np.array(mu), np.array(sigma))
    numerical_ev, err = quad(
        lambda x: x**n * jump_relu(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    # Don't use relative tolerance because the expected value can be very small.
    # The sharpness of ReLU also makes quad have a hard time estimating the error.
    assert abs(numerical_ev - analytic_ev) < 1e-15 + err
    torch_test(partial(jump_relu_poly_ev, n), numerical_ev, mu, sigma, atol=1e-15 + err)
