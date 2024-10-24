from scipy.stats import norm
import numpy as np


def gelu(x):
    """Gaussian Error Linear Unit (GELU) activation function"""
    return x * norm.cdf(x)


def gelu_prime(x):
    """Derivative of GELU(x)"""
    return norm.cdf(x) + x * norm.pdf(x)


def gelu_dbl_prime(x):
    """Second derivative of GELU(x)"""
    return (2 - x ** 2) * norm.pdf(x)


def gelu_ev(mu, sigma):
    """Expected value of GELU(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)
    return mu * norm.cdf(mu / denom) + (sigma ** 2 / denom) * norm.pdf(mu / denom)


def gelu_prime_ev(mu, sigma):
    """Expected value of GELU'(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)

    inner = mu / denom - mu * sigma ** 2 / denom ** 3
    return norm.cdf(mu / denom) + norm.pdf(mu / denom) * inner


def gelu_dbl_prime_ev(mu, sigma):
    """Expected value of GELU''(x) under N(mu, sigma)"""
    denom = np.sqrt(1 + sigma ** 2)

    inner1 = 1 / denom - 3 * mu ** 2 / denom ** 3
    inner2 = -3 * mu / denom ** 3 + 9 * mu ** 3 / denom ** 5
    return norm.cdf(mu / denom) + norm.pdf(mu / denom) * inner1 + norm.pdf(mu / denom) * inner2