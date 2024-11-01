import math

from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
import numpy as np

def relu(x):
    """Rectified Linear Unit (ReLU) activation function"""
    return np.maximum(0, x)


def relu_prime(x):
    """Derivative of ReLU(x)"""
    return np.where(x > 0, 1.0, 0.0)


def relu_ev(mu, sigma):
    """Expected value of RELU(x) under N(mu, sigma)"""
    return mu * norm.cdf(mu / sigma) + sigma * norm.pdf(mu / sigma)


def relu_prime_ev(mu, sigma):
    """Expected value of RELU'(x) under N(mu, sigma)"""
    return norm.cdf(mu / sigma)


def relu_squared_ev(mu, sigma):
    """Expected value of RELU(x)^2 under N(mu, sigma)"""
    loc = mu / sigma
    return sigma ** 2 * (loc ** 2 + 1) * norm.cdf(loc) + mu * sigma * norm.pdf(loc)


def relu_poly_ev(n: int, mu: ArrayLike, sigma: ArrayLike) -> NDArray:
    """
    Compute E[x^n * ReLU(x)] analytically where x ~ N(mu, sigma^2)

    Parameters:
    n     : int, the exponent n in x^n * ReLU(x)
    mu    : ArrayLike, the mean(s) of the normal distribution(s)
    sigma : ArrayLike, the standard deviation(s) of the normal distribution(s)

    Returns:
    result : NDArray, the computed expected value(s)
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    k = mu / sigma
    expected_value = np.zeros_like(mu)

    # Precompute standard normal PDF and CDF at k
    phi_k = norm.pdf(k)  # PDF of standard normal at k
    Phi_k = norm.cdf(k)  # CDF of standard normal at k

    # Compute M_0 and M_1
    M = [Phi_k, phi_k]

    # Compute higher-order M_k recursively
    for i in range(2, n + 2):
        M.append(k ** (i - 1) * phi_k + (i - 1) * M[i - 2])

    # Sum over k from 0 to n+1
    for i in range(n + 2):
        binom_coeff = math.comb(n + 1, i)
        mu_power = mu ** (n + 1 - i)
        sigma_power = sigma ** i
        expected_value += binom_coeff * mu_power * sigma_power * M[i]

    return expected_value