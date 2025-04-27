import numpy as np
import torch
from pathlib import Path
from typing import Tuple

from polyapprox.crosscoders import MLP, CrossCoderTrainer, CrossCoderTrainerTrainingArgs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def main():
    W1, b1, W2, b2 = get_tensors()

    mlp = MLP(W1, b1, W2, b2, act="gelu")

    args = CrossCoderTrainerTrainingArgs(
        num_features=4096,
        target_l0=32,
        num_steps=1_000,
        learning_rate=1e-3,
        log_frequency=10,
        num_gamma_rows=1024
    )

    trainer = CrossCoderTrainer(mlp, args)

    trainer.train()

def get_tensors() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    path = Path(__file__).parent.parent / "tensors" / "gelu-1l-state-dict.pt"

    all_tensors = torch.load(path)

    names = ["blocks.0.mlp.W_in", "blocks.0.mlp.b_in", "blocks.0.mlp.W_out", "blocks.0.mlp.b_out"]

    return_list = []
    for name in names:
        if "W" in name:
            return_list.append(all_tensors[name].T.to(device))
        else:
            return_list.append(all_tensors[name].to(device))

    return_tuple = tuple(return_list)

    return return_tuple


if __name__ == "__main__":
    main()
