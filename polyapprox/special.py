import math
import torch
from torch import Tensor

from scipy import special

def ncdf_t(x, df, delta):
    """
    Calculate the CDF of the noncentral t-distribution.
    
    Parameters:
    -----------
    x : torch.Tensor
        The point at which to evaluate the CDF
    df : float or torch.Tensor
        Degrees of freedom
    nc : float or torch.Tensor
        Noncentrality parameter
        
    Returns:
    --------
    torch.Tensor
        The CDF value(s)
    """
    # Make sure we weren't expecting a gradient
    assert not x.requires_grad and not df.requires_grad and not delta.requires_grad

    # Convert inputs to NumPy for SciPy calculation
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    df_np = df.detach().cpu().numpy() if isinstance(df, torch.Tensor) else df
    delta_np = delta.detach().cpu().numpy() if isinstance(delta, torch.Tensor) else delta
    
    # Use SciPy's implementation
    result_np = special.nctdtr(df_np, delta_np, x_np)
    
    # Convert back to PyTorch tensor
    result = torch.tensor(result_np, dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float32)
    
    # If input was on GPU, move result there too
    if isinstance(x, torch.Tensor) and x.is_cuda:
        result = result.to(x.device)
        
    return result