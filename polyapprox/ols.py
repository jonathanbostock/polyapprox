from dataclasses import dataclass
from functools import partial
from typing import Generic

import array_api_compat
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import ArrayType
from .extra import sigmoid, sigmoid_prime, swish, swish_prime
from .gelu import gelu_ev, gelu_prime_ev
from .integrate import bivariate_product_moment, gauss_hermite
from .relu import relu_ev, relu_prime_ev


@dataclass(frozen=True)
class OlsResult(Generic[ArrayType]):
    alpha: ArrayType
    """Intercept of the linear model."""

    beta: ArrayType
    """Coefficients of the linear model."""

    mean: ArrayType
    """Mean of the output distribution."""

    fvu: float | None = None
    """Fraction of variance unexplained, if available.

    Currently only implemented for ReLU activations.
    """

    def __call__(self, x: ArrayType) -> ArrayType:
        """Evaluate the linear model at the given inputs."""
        return x @ self.beta + self.alpha


# Mapping from activation functions to EVs
ACT_TO_EVS = {
    "gelu": gelu_ev,
    "relu": relu_ev,
    "sigmoid": partial(gauss_hermite, sigmoid),
    "swish": partial(gauss_hermite, swish),
    "tanh": partial(gauss_hermite, np.tanh),
}
# Mapping from activation functions to EVs of their derivatives
ACT_TO_PRIME_EVS = {
    "gelu": gelu_prime_ev,
    "relu": relu_prime_ev,
    "sigmoid": partial(gauss_hermite, sigmoid_prime),
    "swish": partial(gauss_hermite, swish_prime),
    "tanh": partial(gauss_hermite, lambda x: 1 - np.tanh(x) ** 2),
}


def ols(
    W1: ArrayType,
    b1: ArrayType,
    W2: ArrayType,
    b2: ArrayType,
    *,
    act: str = "gelu",
    mean: ArrayType | None = None,
    cov: ArrayType | None = None,
    return_fvu: bool = False,
) -> OlsResult[ArrayType]:
    """Ordinary least squares approximation of a single hidden layer MLP.

    Args:
        W1: Weight matrix of the first layer.
        b1: Bias vector of the first layer.
        W2: Weight matrix of the second layer.
        b2: Bias vector of the second layer.
        mean: Mean of the input distribution. If None, the mean is zero.
        cov: Covariance of the input distribution. If None, the covariance is the
            identity matrix.
        return_fvu: Whether to compute the fraction of variance unexplained.
            This is only available for ReLU activations, and can be computationally
            expensive for large networks.
    """
    xp = array_api_compat.array_namespace(W1, b1, W2, b2)

    # Preactivations are Gaussian; compute their mean and standard deviation
    if cov is not None:
        preact_cov = W1 @ cov @ W1.T
        cross_cov = cov @ W1.T
    else:
        preact_cov = W1 @ W1.T
        cross_cov = W1.T

    preact_mean = b1
    preact_var = xp.diag(preact_cov)
    preact_std = xp.sqrt(preact_var)
    if mean is not None:
        preact_mean = preact_mean + W1 @ mean

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
        beta = xp.linalg.solve(cov, cross_cov)
    else:
        beta = cross_cov

    alpha = output_mean
    if mean is not None:
        alpha -= beta.T @ mean

    # For ReLU, we can compute the covariance matrix of the activations, which is
    # useful for computing the fraction of variance unexplained in closed form.
    if act == "relu" and return_fvu:
        rhos = preact_cov / xp.outer(preact_std, preact_std)

        # Compute the raw second moment matrix of the activations
        act_m2 = bivariate_product_moment(
            0.0,
            0.0,
            rhos,
            mean_x=preact_mean[:, None],
            mean_y=preact_mean[None],
            std_x=preact_std[:, None],
            std_y=preact_std[None],
            unconditional=True,
        )

        # E[MLP(x)^T MLP(x)]
        mlp_scale = xp.trace(W2 @ act_m2 @ W2.T) + 2 * act_mean.T @ W2.T @ b2 + b2 @ b2

        # E[g(x)^T MLP(x)] where g(x) is the linear predictor
        x_moment = cross_cov + (xp.outer(mean, output_mean) if mean is not None else 0)
        inner_prod = xp.trace(beta.T @ x_moment) + alpha.T @ output_mean

        # E[g(x)^T g(x)] where g(x) is the linear predictor
        inner = 2 * mean.T @ beta @ alpha if mean is not None else 0
        lin_scale = xp.trace(beta.T @ cov @ beta) + inner + alpha.T @ alpha

        # Fraction of variance unexplained
        denom = mlp_scale - output_mean @ output_mean
        fvu = (mlp_scale - 2 * inner_prod + lin_scale) / denom
    else:
        fvu = None

    return OlsResult(alpha, beta, output_mean, fvu=fvu)


def glu_mean(
    W: NDArray,
    V: NDArray,
    b1: ArrayLike = 0.0,
    b2: ArrayLike = 0.0,
    *,
    act: str = "sigmoid",
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
