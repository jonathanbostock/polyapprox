from collections import namedtuple
import numpy as np

from .gelu import gelu_ev, gelu_prime_ev


OlsResult = namedtuple('OlsResult', ['alpha', 'beta'])


def gelu_ols(W1, b1, W2) -> OlsResult:
    """Ordinary least squares approximation of a GELU network"""

    # Preactivations are Gaussian with the following parameters
    preact_std = np.linalg.norm(W1, axis=1) # sqrt(diag(W1 @ W1.T))
    preact_mean = b1

    # Cross-covariance between inputs and preactivations
    cross_cov = W1

    # Compute expectation of GELU'(x) for each preactivation
    act_mean = gelu_prime_ev(preact_mean, preact_std)
    beta = W2 @ (cross_cov * act_mean[:, None])

    # Compute expectation of GELU(x) for each preactivation
    act_mean = gelu_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean

    alpha = output_mean # + beta @ mu
    return OlsResult(alpha, beta)