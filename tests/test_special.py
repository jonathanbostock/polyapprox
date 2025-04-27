import pytest
import torch
from scipy.stats import nct

from polyapprox.special import ncdf_t

@pytest.mark.parametrize("x", [0.0, 1.0, 2.0])
@pytest.mark.parametrize("df", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("delta", [0.0, 1.0, 2.0])
def test_ncdf_t(x, df, delta):
    result = ncdf_t(torch.tensor(x), torch.tensor(df), torch.tensor(delta)).type(torch.float64)
    expected = torch.tensor(nct.cdf(x, df, delta))
    assert torch.allclose(result, expected)
