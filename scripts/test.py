import torch

from polyapprox.ols import ols


def main():
    d_input = 5
    d_hidden = 10
    d_output = 2

    W_1 = torch.randn(d_hidden, d_input)
    b_1 = torch.randn(d_hidden)
    W_2 = torch.randn(d_output, d_hidden)
    b_2 = torch.randn(d_output)

    quadratic_term_samples = torch.tensor([0, 1, 2, 3, 4, 5])

    ols_result = ols(W_1, b_1, W_2, b_2, order="quadratic")
    ols_result_subset = ols(W_1, b_1, W_2, b_2, order="quadratic", quadratic_term_samples=quadratic_term_samples)

    print(ols_result.gamma)
    print(ols_result_subset.gamma)


if __name__ == "__main__":
    main()
