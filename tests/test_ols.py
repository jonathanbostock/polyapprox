import numpy as np

from polyapprox.ols import ols


def test_ols_relu():
    d = 10
    nil = np.zeros(d)

    W1 = np.random.randn(d, d)
    W2 = np.random.randn(d, d)

    # When x ~ N(0, 1) and there are no biases, the coefficients take the intuitive
    # form: 0.5 * W2 @ W1
    lin_res = ols(W1, nil, W2, nil, act="relu", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="relu", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.T, 0.5 * W2 @ W1)
    np.testing.assert_allclose(quad_res.beta.T, 0.5 * W2 @ W1)

    # Monte Carlo check that the FVU is below 1
    n = 10_000
    x = np.random.randn(n, d)
    y = np.maximum(0, x @ W1.T) @ W2.T

    # Quadratic FVU should be lower than linear FVU, which should be lower than 1
    lin_fvu = np.square(y - lin_res(x)).sum() / np.square(y).sum()
    quad_fvu = np.square(y - quad_res(x)).sum() / np.square(y).sum()
    assert quad_fvu < lin_fvu < 1

    # Check the trivial case where the activation is the identity
    lin_res = ols(W1, nil, W2, nil, act="identity", order="linear")
    quad_res = ols(W1, nil, W2, nil, act="identity", order="quadratic")
    np.testing.assert_allclose(lin_res.beta.T, W2 @ W1)
    np.testing.assert_allclose(quad_res.beta.T, W2 @ W1)

    assert lin_res.gamma is None and quad_res.gamma is not None
    np.testing.assert_allclose(lin_res.alpha, 0)
    np.testing.assert_allclose(quad_res.alpha, 0)
    np.testing.assert_allclose(quad_res.gamma, 0)
