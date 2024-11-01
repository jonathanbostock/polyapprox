from dataclasses import dataclass
from functools import partial

from numpy.typing import ArrayLike, NDArray
import numpy as np

from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev
from .integrate import bivariate_product_moment, gauss_hermite
from .relu import relu_ev, relu_prime_ev


@dataclass(frozen=True)
class OlsResult:
    alpha: NDArray
    """Intercept of the linear model."""

    beta: NDArray
    """Coefficients of the linear model."""

    mean: NDArray
    """Mean of the output distribution."""

    fvu: float | None = None
    """Fraction of variance unexplained, if available.
    
    Currently only implemented for ReLU activations.
    """

    def __call__(self, x: NDArray) -> NDArray:
        """Evaluate the linear model at the given inputs."""
        return x @ self.beta + self.alpha


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
    b2: NDArray,
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
        preact_cov = W1 @ cov @ W1.T
        cross_cov = cov @ W1.T
    else:
        preact_cov = W1 @ W1.T
        cross_cov = W1.T

    preact_mean = b1.copy()
    preact_var = np.diag(preact_cov)
    preact_std = np.sqrt(preact_var)
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
    cross_cov = (cross_cov * act_mean) @ W2.T

    # Compute expectation of act_fn(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = W2 @ act_mean + b2

    # beta = Cov(x)^-1 Cov(x, f(x))
    if cov is not None:
        beta = np.linalg.solve(cov, cross_cov)
    else:
        beta = cross_cov

    alpha = output_mean
    if mean is not None:
        alpha -= beta.T @ mean

    # For ReLU, we can compute the covariance matrix of the activations, which is
    # useful for computing the fraction of variance unexplained in closed form.
    if act == 'relu':
        rhos = preact_cov / np.outer(preact_std, preact_std)

        # Compute the raw second moment matrix of the activations
        act_m2 = bivariate_product_moment(
            0.0, 0.0, rhos,
            mean_x=preact_mean[:, None],
            mean_y=preact_mean[None],
            std_x=preact_std[:, None],
            std_y=preact_std[None],
            unconditional=True,
        )

        # E[MLP(x)^T MLP(x)]
        mlp_scale = np.trace(W2 @ act_m2 @ W2.T) + 2 * act_mean.T @ W2.T @ b2 + b2 @ b2

        # E[g(x)^T MLP(x)] where g(x) is the linear predictor
        x_moment = cross_cov + (np.outer(mean, output_mean) if mean is not None else 0)
        inner_prod = np.trace(beta.T @ x_moment) + alpha.T @ output_mean

        # E[g(x)^T g(x)] where g(x) is the linear predictor
        inner = 2 * mean.T @ beta @ alpha if mean is not None else 0
        lin_scale = np.trace(beta.T @ cov @ beta) + inner + alpha.T @ alpha

        # Fraction of variance unexplained
        fvu = (mlp_scale - 2 * inner_prod + lin_scale) / mlp_scale
    else:
        fvu = None

    return OlsResult(alpha, beta, output_mean, fvu=fvu)


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
    # The lemma says that Cov(σ(y_i), z_i) = Cov(y_i, z_i) * E[σ'(y_i)]
    # so we need to compute E[σ'(y_i)] for each i
    act_mean = act_ev(y_mean, y_std)
    output_mean = np.diag(cross_cov) * act_prime_ev(y_mean, y_std) + act_mean * z_mean

    return output_mean
