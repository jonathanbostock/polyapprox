from collections import namedtuple
from functools import partial

from numpy.typing import ArrayLike, NDArray
import numpy as np

from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev
from .hermite import gauss_hermite
from .relu import relu_ev, relu_prime_ev


OlsResult = namedtuple('OlsResult', ['alpha', 'beta'])


# Mapping from activation functions to EVs
ACT_TO_EVS = {
    'gelu': gelu_ev,
    'relu': relu_ev,
    'sigmoid': partial(gauss_hermite, sigmoid),
    'swish': partial(gauss_hermite, swish),
    'tanh': partial(gauss_hermite, np.tanh),
}
# Mapping from activation functions to EVs of their derivatives
ACT_TO_PRIME_EVS = {
    'gelu': gelu_prime_ev,
    'relu': relu_prime_ev,
    'sigmoid': partial(gauss_hermite, sigmoid_prime),
    'swish': partial(gauss_hermite, swish_prime),
    'tanh': partial(gauss_hermite, lambda x: 1 - np.tanh(x)**2),
}


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

    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
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


def glu_mean(
    W: NDArray,
    V: NDArray,
    b1: ArrayLike = 0.0,
    b2: ArrayLike = 0.0,
    *,
    act: str = 'sigmoid',
    mean: NDArray | None = None,
    cov: NDArray | None = None,
):
    """Analytically compute the mean output of a gated linear unit (GLU).

    See "GLU Variants Improve Transformer" <https://arxiv.org/abs/2002.05202>
    by Shazeer (2020) for more details.
    """
    # The network takes the form σ(W @ x + b1) * (V @ x + b2)
    # Let y = W @ x + b1 and z = V @ x + b2
    if cov is not None:
        # Cross-covariance matrix of y and z
        cross_cov = W @ cov @ V.T

        y_std = np.diag(W @ cov @ W.T) ** 0.5
        # z_std = np.diag(V @ cov @ V.T) ** 0.5
    else:
        cross_cov = W @ V.T
        y_std = np.linalg.norm(W, axis=1)
        # z_std = np.linalg.norm(V, axis=1)

    y_mean = np.array(b1)
    if mean is not None:
        y_mean += W @ mean

    z_mean = np.array(b2)
    if mean is not None:
        z_mean += V @ mean
    
    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")
    
    # Apply Stein's lemma to compute
    # E[GLU(x)]_i = E[σ(y_i) * z_i] = Cov(σ(y_i), z_i) + E[σ(y_i)] * E[z_i]
    # The lemma says that Cov(y_i, σ(z_i)) = Cov(y_i, z_i) * E[σ'(y_i)]
    # so we need to compute E[σ'(y_i)] for each i
    act_mean = act_ev(y_mean, y_std)
    output_mean = np.diag(cross_cov) * act_prime_ev(y_mean, y_std) + act_mean * z_mean

    return output_mean
