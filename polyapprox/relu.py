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