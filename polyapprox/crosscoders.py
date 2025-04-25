import numpy as np
import torch
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
from typing import Literal, Optional

from dataclasses import dataclass
from typing import Dict

from .backends import norm_cdf
from .ols import ols

@dataclass
class MLPData:
    W1: torch.Tensor
    b1: torch.Tensor
    W2: torch.Tensor
    b2: torch.Tensor

    def __post_init__(self):
        self.params = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        self.d_input = self.W1.shape[1]
        self.d_hidden = self.W1.shape[0]
        self.d_output = self.W2.shape[0]

        self.dims = {
            "d_input": self.d_input,
            "d_hidden": self.d_hidden,
            "d_output": self.d_output,
        }

class CrossCoder(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, d_output: int):
        super().__init__()

        self.W1 = nn.Parameter(torch.randn(d_hidden, d_input))
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.W2 = nn.Parameter(torch.randn(d_output, d_hidden))
        self.b2 = nn.Parameter(torch.zeros(d_output))

        self.params = nn.ParameterDict({
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
        """

        preactivations = x @ self.W1 + self.b1
        active_neurons = (preactivations > 1).float()
        activations = preactivations * active_neurons

        return {
            "preactivations": preactivations,
            "active_features": active_neurons,
            "activations": activations,
            "outputs": activations @ self.W2 + self.b2,
        }
    
    def estimate_l0(self) -> torch.Tensor:
        """Estimate the L0 norm of the model."""

        W1_row_mags = torch.sqrt(torch.sum(self.W1 ** 2, dim=1))
        b1_relative_mags = self.b1 / W1_row_mags
        estimated_l0 = torch.sum(norm_cdf(b1_relative_mags))
        return estimated_l0

@dataclass(frozen=True)
class CrossCoderTrainerTrainingArgs:
    num_features: int
    target_l0: int
    num_steps: int
    learning_rate: float
    num_gamma_rows: Optional[int] = None

class CrossCoderTrainer:
    def __init__(
            self, 
            mlp: MLPData,
            optimizer: Literal["adamw"],
            args: CrossCoderTrainerTrainingArgs,
            seed: int = 42,
            ) -> None:
        

        self.mlp = mlp
        self.crosscoder = CrossCoder(mlp.d_input, args.num_features, mlp.d_output)
        self.optimizer = optimizer
        self.args = args
        
        if optimizer == "adamw":
            self.optimizer = AdamW(self.crosscoder.params.values(), lr=args.learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")
        
        self.randomizer = np.random.default_rng(seed)

    def train(self) -> None:

        for i in range(self.args.num_steps):

            if self.args.num_gamma_rows is not None:
                quadratic_term_samples = torch.from_numpy(self.randomizer.choice(
                    range(self.mlp.d_input * (self.mlp.d_input + 1) // 2),
                    size=self.args.num_gamma_rows,
                    replace=False,
                ))
            else:
                quadratic_term_samples = None


            with torch.no_grad():
                mlp_data = ols(**self.mlp.params, act="gelu", order="quadratic", quadratic_term_samples=None)

            crosscoder_data = ols(**self.crosscoder.params, act="jump_relu", order="quadratic", quadratic_term_samples=None)

            self.optimizer.zero_grad()

            crosscoder_loss = F.mse_loss(crosscoder_data.coefficients(), mlp_data.coefficients())
            crosscoder_loss_value = crosscoder_loss.item()
            l0_loss = (self.crosscoder.estimate_l0() / self.args.target_l0 - 1) ** 2
            l0_loss_value = l0_loss.item()

            loss = crosscoder_loss + l0_loss
            loss_value = loss.item()
            loss.backward()
            self.optimizer.step()

            print(f"Step {i}: Loss = {loss_value}, Crosscoder Loss = {crosscoder_loss_value}, L0 Loss = {l0_loss_value}")
