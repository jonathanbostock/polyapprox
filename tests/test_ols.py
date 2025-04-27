import numpy as np
import torch
import pytest
import statsmodels.api as sm
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS

from polyapprox.gelu import gelu
from polyapprox.jump_relu import jump_relu
from polyapprox.relu import relu
from polyapprox.ols import ols

precision = 1e-9
torch.set_default_dtype(torch.float64)

ols_monte_carlo_tolerance = {
    "relu": 0.0,
    "gelu": 0.0,
    "jump_relu": 0.05,
}

def test_ols_relu():
    torch.manual_seed(0)

    d = 10
    nil = torch.zeros(d)

    W1 = torch.randn(d, d)
    W2 = torch.randn(d, d)

    # When x ~ N(0, 1) and there are no biases, the coefficients take the intuitive
    # form: 0.5 * W2 @ W1
    lin_res = ols(W1, nil, W2, nil, act="relu", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="relu", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.T, 0.5 * W2 @ W1, atol=1e-6)
    np.testing.assert_allclose(quad_res.beta.T, 0.5 * W2 @ W1, atol=1e-6)

    # Monte Carlo check that the FVU is below 1
    n = 10_000
    x = torch.randn(n, d)
    y = relu(x @ W1.T) @ W2.T

    # Quadratic FVU should be lower than linear FVU, which should be lower than 1
    lin_fvu = (y - lin_res(x)).square().sum() / y.square().sum()
    quad_fvu = (y - quad_res(x)).square().sum() / y.square().sum()
    assert quad_fvu < lin_fvu < 1

    # Check the trivial case where the activation is the identity
    lin_res = ols(W1, nil, W2, nil, act="identity", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="identity", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.detach().cpu().numpy().T, (W2 @ W1).detach().cpu().numpy(), atol=precision)
    np.testing.assert_allclose(quad_res.beta.detach().cpu().numpy().T, (W2 @ W1).detach().cpu().numpy(), atol=precision)

    assert lin_res.gamma is None and quad_res.gamma is not None
    np.testing.assert_allclose(lin_res.alpha.detach().cpu().numpy(), 0, atol=precision)
    np.testing.assert_allclose(quad_res.alpha.detach().cpu().numpy(), 0, atol=precision)
    np.testing.assert_allclose(quad_res.gamma.detach().cpu().numpy(), 0, atol=precision)


@pytest.mark.parametrize("act", ["gelu", "relu", "jump_relu"])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_ols_monte_carlo(act: str, k: int):
    # Determinism
    torch.manual_seed(0)
    np.random.seed(0)

    # Choose activation function
    match act:
        case "gelu":
            act_fn = gelu
        case "relu":
            act_fn = relu
        case "jump_relu":
            act_fn = jump_relu
        case _:
            raise ValueError(f"Unknown activation function: {act}")

    # Implement the MLP
    def mlp(x, W1, b1, W2, b2):
        return act_fn(x @ W1.T + b1) @ W2.T + b2

    d_in = 10
    d_inner = 1_000
    d_out = 1

    # Construct a random Gaussian mixture with k components
    # random psd matrix d_in x d_in
    A = torch.randn(k, d_in, d_in) / np.sqrt(d_in)
    cov_x = A @ A.mT
    mu_x = torch.randn(k, d_in)

    W1 = torch.randn(d_inner, d_in) / np.sqrt(d_in)
    W2 = torch.randn(d_out, d_inner) / np.sqrt(d_inner)
    b1 = torch.randn(d_inner) / np.sqrt(d_in)
    b2 = torch.randn(d_out)

    # Compute analytic coefficients
    analytic = ols(W1, b1, W2, b2, act=act, cov=cov_x.squeeze(), mean=mu_x.squeeze())

    # Generate Monte Carlo data
    x = np.concatenate(
        [mvn.rvs(mean=mu_x[i], cov=cov_x[i], size=10_000) for i in range(k)]
    )
    y = mlp(torch.from_numpy(x), W1, b1, W2, b2)

    # Use statsmodels to approximate the coefficients
    X = sm.add_constant(x)
    empirical = OLS(y.numpy(), X).fit()

    # Check that analytic coefficients are within the confidence interval
    lo, hi = empirical.conf_int(0.01).T
    analytic_beta = analytic.beta.squeeze().numpy()

    tol = ols_monte_carlo_tolerance[act]

    assert lo[0] - tol < analytic.alpha.numpy() < hi[0] + tol
    assert np.all((lo[1:] - tol < analytic_beta) & (analytic_beta < hi[1:] + tol))
