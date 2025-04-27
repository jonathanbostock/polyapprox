from functools import partial

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.jump_relu import jump_relu_ev, jump_relu_poly_ev, jump_relu_prime_ev

# NB: This is mostly a copy of test_relu.py, with the ReLU replaced with JumpReLU
# The main difference is that we have added 1e-14 to the error tolerance, because
# the sharpness of JumpReLU makes quad have a hard time estimating the error.
# Frankly. 1e-14 error is good enough for anyone.

# Disable the dirac delta term in the expected value of the derivative
jump_relu_prime_ev = partial(jump_relu_prime_ev, include_dirac_delta_term=False)

precision = 1e-9
torch.set_default_dtype(torch.float64)

def jump_relu_numpy(x):
    """JumpReLU(x) activation function, with theta = 1.0"""
    return np.where(x > 1.0, x, 0)


def jump_relu_prime_numpy(x):
    """Derivative of JumpReLU(x) with theta = 1.0"""
    return np.where(x > 1.0, 1.0, 0.0)



def torch_test(f, target, *args, atol=1e-12):
    res = f(*map(lambda x: torch.tensor(x, dtype=torch.double), args))
    assert abs(res - target) < atol + torch.finfo(torch.double).eps


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_jump_relu_evs(mu, sigma):
    analytic_ev = jump_relu_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: jump_relu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=precision + err)
    torch_test(jump_relu_ev, numerical_ev, mu, sigma, atol=precision + err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_prime_evs(mu, sigma):
    analytic_ev = jump_relu_prime_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: jump_relu_prime_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=precision + err)
    torch_test(jump_relu_prime_ev, numerical_ev, mu, sigma, atol=precision + err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_jump_relu_poly_evs(n, mu, sigma):
    analytic_ev = jump_relu_poly_ev(n, torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: x**n * jump_relu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    assert abs(numerical_ev - analytic_ev) < precision + err
    torch_test(partial(jump_relu_poly_ev, n), numerical_ev, mu, sigma, atol=precision + err)