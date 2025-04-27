import math
import torch

def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of N(0, 1)"""

    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def norm_pdf(x: torch.Tensor) -> torch.Tensor:
    """Probability density function of N(0, 1)"""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)