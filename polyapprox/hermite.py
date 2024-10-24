from typing import Callable

from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_hermite
import numpy as np


def gauss_hermite(
    f: Callable,
    mu: ArrayLike = 0.0,
    sigma: ArrayLike = 1.0,
    num_points: int = 50,
) -> NDArray:
    """
    Compute E[f(x)] where x ~ N(mu, sigma^2) using Gauss-Hermite quadrature.

    Parameters:
    - mu: array-like, means
    - sigma: array-like, standard deviations
    - num_points: int, number of quadrature points

    Returns:
    - expectations: array-like, E[f(x)] for each (mu, sigma)
    """
    # Obtain Gauss-Hermite nodes and weights
    nodes, weights = roots_hermite(num_points)  # Nodes: z_i, Weights: w_i

    # Reshape for broadcasting
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    # See example in https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
    grid = mu[:, None] + sigma[:, None] * np.sqrt(2) * nodes

    # Compute the weighted sum
    return np.dot(f(grid), weights) / np.sqrt(np.pi)