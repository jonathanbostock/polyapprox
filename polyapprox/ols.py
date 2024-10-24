from collections import namedtuple
from functools import partial

from numpy.typing import NDArray
import numpy as np

from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev
from .hermite import gauss_hermite
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

        cross_cov = cov @ W1.T
    else:
        preact_std = np.linalg.norm(W1, axis=1) # sqrt(diag(W1 @ W1.T))
        cross_cov = W1.T

    preact_mean = b1.copy()
    if mean is not None:
        preact_mean += W1 @ mean

    match act:
        case 'gelu':
            act_ev = gelu_ev
            act_prime_ev = gelu_prime_ev
        case 'relu':
            act_ev = relu_ev
            act_prime_ev = relu_prime_ev
        case 'sigmoid':
            act_ev = partial(gauss_hermite, sigmoid)
            act_prime_ev = partial(gauss_hermite, sigmoid_prime)
        case 'swish':
            act_ev = partial(gauss_hermite, swish)
            act_prime_ev = partial(gauss_hermite, swish_prime)
        case 'tanh':
            act_ev = partial(gauss_hermite, np.tanh)
            act_prime_ev = partial(gauss_hermite, lambda x: 1 - np.tanh(x)**2)
        case _:
            raise ValueError(f"Unknown activation function: {act}")

    # Apply Stein's lemma to compute cross-covariance of the input
    # with the activations. We need the expected derivative of the
    # activation function with respect to the preactivation.
    act_mean = act_prime_ev(preact_mean, preact_std)
    beta = (cross_cov * act_mean) @ W2.T

    # Compute expectation of GELU(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean + b2

    # beta = Cov(x)^-1 Cov(x, f(x))
    if cov is not None:
        beta = np.linalg.solve(cov, beta)

    alpha = output_mean
    if mean is not None:
        alpha -= beta.T @ mean

    return OlsResult(alpha, beta)