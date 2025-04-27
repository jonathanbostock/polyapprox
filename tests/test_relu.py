from functools import partial

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.relu import relu_ev, relu_poly_ev, relu_prime_ev

torch.set_default_dtype(torch.float64)

precision = 1e-9

def relu(x):
    """Rectified Linear Unit (ReLU) activation function"""
    return torch.maximum(0, x)

def relu_numpy(x):
    return np.maximum(0, x)

def relu_prime(x):
    """Derivative of ReLU(x)"""
    return torch.where(x > 0, 1.0, 0.0)

def relu_prime_numpy(x):
    return np.where(x > 0, 1.0, 0.0)


def torch_test(f, target, *args, atol=1e-12):
    res = f(*map(lambda x: torch.tensor(x, dtype=torch.double), args))
    assert abs(res - target) < atol + torch.finfo(torch.double).eps


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_evs(mu, sigma):
    analytic_ev = relu_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: relu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev.detach().cpu().numpy(), atol=precision + err)
    torch_test(relu_ev, numerical_ev, mu, sigma, atol=precision + err)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_prime_evs(mu, sigma):
    analytic_ev = relu_prime_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: relu_prime_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev.detach().cpu().numpy(), atol=precision + err)
    torch_test(relu_prime_ev, numerical_ev, mu, sigma, atol=precision + err)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_relu_poly_evs(n, mu, sigma):
    analytic_ev = relu_poly_ev(n, torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: x**n * relu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    # Don't use relative tolerance because the expected value can be very small.
    # The sharpness of ReLU also makes quad have a hard time estimating the error.
    assert abs(numerical_ev - analytic_ev.detach().cpu().numpy()) < precision + err
    torch_test(partial(relu_poly_ev, n), numerical_ev, mu, sigma, atol=precision + err)
