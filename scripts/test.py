import torch

from polyapprox.ols import ols


def main():
    W_1 = torch.randn(20, 10)
    b_1 = torch.randn(20)
    W_2 = torch.randn(5, 20)
    b_2 = torch.randn(5)

    ols_result = ols(W_1, b_1, W_2, b_2, activation="gelu", order="quadratic")
    print(ols_result)


if __name__ == "__main__":
    main()
