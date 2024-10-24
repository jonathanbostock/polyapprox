from scipy.special import expit as sigmoid


def swish(x):
    return x * sigmoid(x)

def swish_prime(x):
    return sigmoid(x) + swish(x) - sigmoid(x) * swish(x)