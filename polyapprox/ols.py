from collections import namedtuple

from numpy.typing import NDArray
import numpy as np

from .gelu import gelu_ev, gelu_prime_ev
from .relu import relu_ev, relu_prime_ev


OlsResult = namedtuple('OlsResult', ['alpha', 'beta'])


def ols(
    W1: NDArray, 
    b1: NDArray,
    W2: NDArray,
    b2 = 0.0,
    *,
    act: str = 'gelu',
    mean: NDArray | None = None,
    cov: NDArray | None = None,
) -> OlsResult:
    """Ordinary least squares approximation of a single hidden layer MLP.

    Args:
        W1: Weight matrix of the first layer.
        b1: Bias vector of the first layer.
        W2: Weight matrix of the second layer.
        b2: Bias vector of the second layer.
        mean: Mean of the input distribution. If None, the mean is zero.
        cov: Covariance of the input distribution. If None, the covariance is the
            identity matrix.
    """
    # Preactivations are Gaussian; compute their mean and standard deviation
    if cov is not None:
        preact_var = np.diag(W1 @ cov @ W1.T)
        preact_std = np.sqrt(preact_var)
    else:
        preact_std = np.linalg.norm(W1, axis=1) # sqrt(diag(W1 @ W1.T))

    preact_mean = b1
    if mean is not None:
        preact_mean += W1 @ mean

    # Cross-covariance between inputs and preactivations
    cross_cov = W1

    match act:
        case 'gelu':
            act_ev = gelu_ev
            act_prime_ev = gelu_prime_ev
        case 'relu':
            act_ev = relu_ev
            act_prime_ev = relu_prime_ev
        case _:
            raise ValueError(f"Unknown activation function: {act}")

    # Compute expectation of GELU'(x) for each preactivation
    act_mean = act_prime_ev(preact_mean, preact_std)
    beta = W2 @ (cross_cov * act_mean[:, None])

    # Compute expectation of GELU(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean + b2

    alpha = output_mean
    if mean is not None:
        alpha -= beta @ mean

    return OlsResult(alpha, beta)