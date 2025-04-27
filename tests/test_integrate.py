import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from scipy.integrate import dblquad, quad
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

from polyapprox.integrate import (
    gauss_hermite,
    isserlis,
)

def gelu_numpy(x):
    return x * norm.cdf(x)

def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))

def swish_numpy(x):
    return x * sigmoid_numpy(x)

@given(st.floats(-4, 4), st.floats(0.1, 10))
def test_gauss_hermite(mu, sigma):
    funcs = [gelu_numpy, sigmoid_numpy, swish_numpy, np.tanh]

    for f in funcs:
        numerical, err = quad(
            lambda x: f(x) * norm.pdf(x, loc=mu, scale=sigma), -np.inf, np.inf
        )

        analytical = gauss_hermite(f, torch.tensor(mu), torch.tensor(sigma), num_points=5000)
        assert abs(numerical - analytical) < err + np.finfo(float).eps

        # Test PyTorch backend
        mu_torch = torch.tensor(mu, dtype=torch.double)
        sigma_torch = torch.tensor(sigma, dtype=torch.double)
        analytical_torch = gauss_hermite(f, mu_torch, sigma_torch, num_points=5000)
        assert abs(numerical - analytical_torch) < err + torch.finfo(torch.double).eps


def test_isserlis():
    B = np.random.randn(2, 2) / np.sqrt(2)
    K = B @ B.T

    analytical = isserlis(torch.tensor(K), [0] * 4 + [1] * 6)
    numerical, error = dblquad(
        lambda y, x: x**4 * y**6 * mvn.pdf([x, y], cov=K),
        -np.inf,
        np.inf,
        -np.inf,
        np.inf,
    )
    np.testing.assert_allclose(numerical, analytical, atol=error)
