import pytest
import torch
from scipy.stats import nct as scipy_nct
from scipy.special import gamma as scipy_gamma  
from polyapprox.special import ncdf_t, gamma

@pytest.mark.parametrize("x", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("df", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("delta", [0.0, 1.0, 2.0])
def test_ncdf_t(x, df, delta):
    result = ncdf_t(torch.tensor(x), torch.tensor(df), torch.tensor(delta)).type(torch.float64)
    expected = torch.tensor(scipy_nct.cdf(x, df, delta))
    assert torch.allclose(result, expected)

@pytest.mark.parametrize("x", [0.0, 1.0, 2.0])
def test_gamma(x):
    result = gamma(torch.tensor(x)).type(torch.float64)
    expected = torch.tensor(scipy_gamma(x))
    assert torch.allclose(result, expected)
