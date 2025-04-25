import torch

from polyapprox.crosscoder import MLPData, CrossCoderTrainer, CrossCoderTrainerTrainingArgs
from polyapprox.ols import ols

def main():
    d_input = 32
    d_hidden = d_input * 4
    d_output = d_input

    W_1 = torch.randn(d_hidden, d_input)
    b_1 = torch.randn(d_hidden)
    W_2 = torch.randn(d_output, d_hidden)
    b_2 = torch.randn(d_output)

    mlp = MLPData(W_1, b_1, W_2, b_2)

    args = CrossCoderTrainerTrainingArgs(
        num_features=1024,
        target_l0=8,
        num_steps=1_000,
        learning_rate=1e-2
    )

    trainer = CrossCoderTrainer(mlp, "adamw", args)

    trainer.train()
if __name__ == "__main__":
    main()
