from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import array_api_compat
import numpy as np
import torch

from torch import Tensor
from .extra import (
    id_poly_ev,
    sigmoid,
    sigmoid_poly_ev,
    sigmoid_prime,
    swish,
    swish_poly_ev,
    swish_prime,
)
from .gelu import gelu_ev, gelu_poly_ev, gelu_prime_ev
from .integrate import (
    gauss_hermite,
    master_theorem,
)
from .relu import relu_ev, relu_poly_ev, relu_prime_ev


@dataclass(frozen=True)
class OlsResult:
    alpha: Tensor
    """Intercept of the linear model."""

    beta: Tensor
    """Coefficients of the linear model."""

    gamma: Tensor | None = None
    """Coefficients for second-order interactions, if available."""

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the linear model at the given inputs."""
        xp = array_api_compat.get_namespace(x)
        y = x @ self.beta + self.alpha

        if self.gamma is not None:
            a = xp.einsum("bi,hij->bhj", x, self.unpack_gamma())
            y += xp.einsum("bj,bhj->bh", x, a)

        return y

    def unpack_gamma(self) -> Tensor:
        """Unpack gamma into a full stack of matrices."""
        if self.gamma is None:
            raise ValueError("No second-order interactions available")

        d_input, d_out = self.beta.shape
        xp = array_api_compat.get_namespace(self.gamma)

        rows, cols = map(xp.asarray, np.tril_indices(d_input))
        gamma = xp.zeros((d_out, d_input, d_input), dtype=self.gamma.dtype)
        gamma[:, rows, cols] = self.gamma
        return 0.5 * (gamma + xp.permute_dims(gamma, (0, 2, 1)))


# Mapping from activation functions to EVs
ACT_TO_EVS = {
    "gelu": gelu_ev,
    "identity": lambda mu, sigma: mu,
    "relu": relu_ev,
    "sigmoid": partial(gauss_hermite, sigmoid),
    "swish": partial(gauss_hermite, swish),
    "tanh": partial(gauss_hermite, np.tanh),
}
# Mapping from activation functions to EVs of their derivatives
ACT_TO_PRIME_EVS = {
    "gelu": gelu_prime_ev,
    "identity": lambda mu, sigma: np.ones_like(mu),
    "relu": relu_prime_ev,
    "sigmoid": partial(gauss_hermite, sigmoid_prime),
    "swish": partial(gauss_hermite, swish_prime),
    "tanh": partial(gauss_hermite, lambda x: 1 - np.tanh(x) ** 2),
}
ACT_TO_POLY_EVS = {
    "gelu": gelu_poly_ev,
    "identity": id_poly_ev,
    "relu": relu_poly_ev,
    "sigmoid": sigmoid_poly_ev,
    "swish": swish_poly_ev,
}


def ols(
    W1: Tensor,
    b1: Tensor,
    W2: Tensor,
    b2: Tensor,
    *,
    act: str = "gelu",
    mean: Tensor | None = None,
    cov: Tensor | None = None,
    order: Literal["linear", "quadratic"] = "linear",
    quadratic_term_samples: Optional[Tensor] = None
) -> OlsResult:
    """Ordinary least squares approximation of a single hidden layer MLP.

    Args:
        W1: Weight matrix of the first layer.
        b1: Bias vector of the first layer.
        W2: Weight matrix of the second layer.
        b2: Bias vector of the second layer.
        mean: Mean of the input distribution. Can include a batch dimension, denoting
            a Gaussian mixture with the specified means. If None, the mean is zero.
        cov: Covariance of the input distribution. Can include a batch dimension,
            denoting a Gaussian mixture with the specified covariance matrices. If
            None, the covariance is the identity matrix.
    """
    d_input = W1.shape[1]

    num_quadratic_terms = (d_input * (d_input + 1)) // 2
    if quadratic_term_samples is not None:
        assert quadratic_term_samples.shape[0] <= num_quadratic_terms

    # Preactivations are Gaussian; compute their mean and standard deviation
    if cov is not None:
        preact_cov = (cov @ W1.T).transpose(-1, -2) @ W1  # Supports batches
        cross_cov = cov @ W1.T
    else:
        preact_cov = W1 @ W1.T
        cross_cov = W1.T

    preact_mean = b1
    preact_var = preact_cov.diagonal(dim1=-2, dim2=-1)
    preact_std = preact_var.sqrt()
    if mean is not None:
        preact_mean = preact_mean + mean @ W1.T

    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")

    # Apply Stein's lemma to compute cross-covariance of the input
    # with the activations. We need the expected derivative of the
    # activation function with respect to the preactivation.
    act_prime_mean = act_prime_ev(preact_mean, preact_std)
    output_cross_cov = (cross_cov * act_prime_mean[..., None, :]) @ W2.T

    # Compute expectation of act_fn(x) for each preactivation
    act_mean = act_ev(preact_mean, preact_std)
    output_mean = act_mean @ W2.T + b2

    # Law of total covariance
    if cov is not None and cov.ndim > 2:
        cov = cov.mean(axis=0)  # type: ignore

    # Average over mixture components if necessary
    if mean is not None and mean.ndim > 1:
        avg_mean = mean.mean(axis=0)  # type: ignore
        avg_output = output_mean.mean(axis=0)

        # Add the covariance of the means to the covariance matrix
        extra_cov = mean.T @ mean / len(mean) - torch.einsum("...i,...j->...ij", avg_mean, avg_mean)
        cov = cov + extra_cov if cov is not None else extra_cov + torch.eye(d_input)

        # Add the cross-covariance of the means to the cross-covariance matrix
        extra_xcov = mean.T @ output_mean / len(mean) - xp.outer(avg_mean, avg_output)
        output_cross_cov = output_cross_cov.mean(axis=0) + extra_xcov

        mean = avg_mean
        output_mean = avg_output

    # beta = Cov(x)^-1 Cov(x, f(x))
    if cov is not None:
        beta = torch.linalg.solve(cov, output_cross_cov)
    else:
        beta = output_cross_cov

    alpha = output_mean
    if mean is not None:
        alpha -= mean @ beta

    if order == "quadratic":
        # Get indices to all unique pairs of input dimensions
        rows, cols = map(lambda x: torch.from_numpy(x).to(torch.int32), np.tril_indices(d_input))

        # TODO: Support non-zero means and non-diagonal covariances
        assert cov is None and mean is None
        sigma = torch.eye(d_input)
        mu = torch.zeros(d_input)

        # "Expand" our covariance and cross-covariance matrices into a batch of 2x2
        # and 2x1 matrices, respectively, to apply the master theorem. Each matrix
        # corresponds to a pairing of two input dimensions. We do a similar thing
        # for the means, which are 1D vectors.
        expanded_cov = torch.stack(
            [
                torch.stack([sigma[rows, rows], sigma[rows, cols]]),
                torch.stack([sigma[cols, rows], sigma[cols, cols]]),
            ]
        ).transpose(0,-1)
        expanded_xcov = torch.stack(
            [
                cross_cov[rows],
                cross_cov[cols],
            ]
        ).transpose(0,-1)
        expanded_mean = torch.stack([mu[rows], mu[cols]]).transpose(0,-1)

        # x_squared_terms = (d_input * (d_input + 1)) // 2
        # expanded_cov has shape (x_squared_terms, 2, 2)
        # expanded_xcov has shape (d_hidden, x_squared_terms, 2)
        # expanded_mean has shape (x_squared_terms, 2)

        # If we are sampling a subset of the quadratic terms, we need to
        # select the corresponding rows and columns from the covariance and
        # cross-covariance matrices, and the corresponding entries from the
        # mean vector.
        if quadratic_term_samples is not None:
            expanded_cov = expanded_cov[quadratic_term_samples, :, :]
            expanded_xcov = expanded_xcov[:, quadratic_term_samples, :]
            expanded_mean = expanded_mean[quadratic_term_samples, :]

        coefs = master_theorem(
            # Add an extra singleton dimension so that we can broadcast across all the
            # pairings of input dimensions
            mu_x=preact_mean[..., None],
            var_x=preact_var[..., None],
            cov_y=expanded_cov,
            xcov=expanded_xcov,
            mu_y=expanded_mean,
        )

        # Compute univariate integrals
        try:
            poly_ev = ACT_TO_POLY_EVS[act]
        except KeyError:
            raise ValueError(f"Quadratic not implemented for activation: {act}")

        quad = poly_ev(2, preact_mean, preact_std)
        lin = poly_ev(1, preact_mean, preact_std)
        const = poly_ev(0, preact_mean, preact_std)
        E_gy_x1x2 = (
            coefs[0] * quad[:, None]
            + coefs[1] * lin[:, None]
            + coefs[2] * const[:, None]
        )

        # TODO: Make this actually work for nontrivial mean and cov
        # Compute the mean and covariance matrix of the input features
        # (products of potentially non-central jointly Gaussian variables)
        # feature_mean = noncentral_isserlis(expanded_cov, expanded_mean)

        # Where rows == cols, E[x * y] = E[x^2] = 1
        # Where rows != cols, E[x * y] = E[x] * E[y] = 0
        feature_mean = (rows == cols).to(W2.dtype)

        if quadratic_term_samples is not None:
            feature_mean = feature_mean[quadratic_term_samples]

        # Where rows == cols, Var[x * y] = Var[x^2] = E[x^4] - E[x^2]^2 = 3 - 1 = 2
        # Where rows != cols, Var[x * y] = E[x^2 * y^2] - E[x * y]^2 = 1 - 0 = 1
        feature_var = 1 + (rows == cols)

        if quadratic_term_samples is not None:
            feature_var = feature_var[quadratic_term_samples]

        quad_xcov = W2 @ (E_gy_x1x2 - torch.einsum("...i,...j->...ij", const, feature_mean))

        gamma = quad_xcov / feature_var

        # adjust constant term
        alpha -= feature_mean.squeeze() @ gamma.mT
    else:
        gamma = None

    return OlsResult(alpha, beta, gamma=gamma)


def glu_ols(
    W: Tensor,
    V: Tensor,
    b1: Tensor,
    b2: Tensor,
    W2: Tensor | None = None,
    *,
    act: str = "sigmoid",
    mean: Tensor | None = None,
    cov: Tensor | None = None,
):
    """Analytically compute the mean output of a gated linear unit (GLU).

    See "GLU Variants Improve Transformer" <https://arxiv.org/abs/2002.05202>
    by Shazeer (2020) for more details.
    """
    if cov is None:
        cov = torch.eye(len(b1))
    if mean is None:
        mean = torch.zeros(len(b1))

    assert cov is not None and mean is not None

    # The network takes the form σ(W @ x + b1) * (V @ x + b2)
    # Let y = W @ x + b1 and z = V @ x + b2
    cov_xz = cov @ V.T
    cov_yx = W @ cov
    cov_yz = torch.diag(W @ cov @ V.T)  # We only need the diagonal
    cov_zz = V @ cov @ V.T

    y_std = torch.diag(W @ cov @ W.T) ** 0.5

    y_mean = b1 + W @ mean
    z_mean = b2 + V @ mean

    try:
        act_ev = ACT_TO_EVS[act]
        act_prime_ev = ACT_TO_PRIME_EVS[act]
        poly_ev = ACT_TO_POLY_EVS[act]
    except KeyError:
        raise ValueError(f"Unknown activation function: {act}")

    # Apply Stein's lemma to compute
    # E[GLU(x)]_i = E[σ(y_i) * z_i] = Cov(σ(y_i), z_i) + E[σ(y_i)] * E[z_i]
    # The lemma says that Cov(σ(y_i), z_i) = Cov(y_i, z_i) * E[σ'(y_i)]
    # so we need to compute E[σ'(y_i)] for each i
    act_mean = act_ev(y_mean, y_std)
    out_mean = cov_yz * act_prime_ev(y_mean, y_std) + act_mean * z_mean
    if W2 is not None:
        out_mean = W2 @ out_mean

    # Apply the master theorem to the integral E[σ(y_i) * z_i * x_j] for each i, j
    # The resulting coefficients have the same shape as cov_xz
    quad = poly_ev(2, y_mean, y_std)
    lin = poly_ev(1, y_mean, y_std)
    const = poly_ev(0, y_mean, y_std)

    var_x = np.broadcast_to(torch.diag(cov)[..., None], cov_xz.shape)
    var_z = np.broadcast_to(torch.diag(cov_zz)[None], cov_xz.shape)
    cov_rest = torch.asarray(
        [
            torch.stack([var_z, cov_xz]),
            torch.stack([cov_xz, var_x]),
        ]
    ).T
    mu_rest = torch.asarray(
        [
            torch.broadcast_to(z_mean[None], cov_xz.shape),
            torch.broadcast_to(mean[..., None], cov_xz.shape),
        ]
    ).T
    xcov = torch.stack(
        [
            torch.broadcast_to(cov_yz[..., None], cov_yx.shape),
            cov_yx,
        ],
        axis=-1,
    )
    coefs = master_theorem(
        mu_x=y_mean[..., None],
        var_x=y_std[..., None] ** 2,
        mu_y=mu_rest,
        cov_y=cov_rest,
        xcov=xcov,
    )
    feature_xcov = (
        coefs[0] * quad[:, None] + coefs[1] * lin[:, None] + coefs[2] * const[:, None]
    )
    beta = (feature_xcov - xp.outer(out_mean, mean)).T
    alpha = out_mean - mean @ beta

    return OlsResult(alpha, beta)
