from functools import partial

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.stats import norm

from polyapprox.gelu import gelu, gelu_prime, gelu_ev, gelu_poly_ev, gelu_prime_ev

torch.set_default_dtype(torch.float64)

precision = 1e-6

def gelu_numpy(x):
    return x * norm.cdf(x)

def gelu_prime_numpy(x):
    return norm.cdf(x) + x * norm.pdf(x)

@given(st.floats(-4, 4))
def test_gelu_numpy(x):
    assert np.isclose(gelu_numpy(x), gelu(torch.tensor(x)).numpy(), atol=precision)

@given(st.floats(-4, 4))
def test_gelu_prime_numpy(x):
    assert np.isclose(gelu_prime_numpy(x), gelu_prime(torch.tensor(x)).numpy(), atol=precision)

def torch_test(f, target, *args, atol=1e-12):
    res = f(*map(lambda x: torch.tensor(x, dtype=torch.double), args))
    np.testing.assert_allclose(res, target, atol=atol)

@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_evs(mu, sigma):
    analytic_ev = gelu_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: gelu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err + precision)
    torch_test(gelu_ev, numerical_ev, mu, sigma, atol=err + precision)


@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_prime_evs(mu, sigma):
    analytic_ev = gelu_prime_ev(torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: gelu_prime_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err + precision)
    torch_test(gelu_prime_ev, numerical_ev, mu, sigma, atol=err + precision)


@given(st.integers(0, 5), st.floats(-4, 4), st.floats(0.1, 10))
def test_gelu_poly_evs(n, mu, sigma):
    analytic_ev = gelu_poly_ev(n, torch.tensor(mu), torch.tensor(sigma))
    numerical_ev, err = quad(
        lambda x: x**n * gelu_numpy(x) * norm.pdf(x, loc=mu, scale=sigma),
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical_ev, analytic_ev, atol=err + precision)
    torch_test(partial(gelu_poly_ev, n), numerical_ev, mu, sigma, atol=err + precision)
