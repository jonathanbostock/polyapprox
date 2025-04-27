# polyapprox
Closed-form polynomial approximations to neural networks.
Repo forked and pared down from EleutherAI.
# crosscoders
Using the polynomial approximations to train crosscoders, by taking the MSE between the polynomial approximations of the crosscoder and the MLP layer and using this as a training loss.
Crosscoders use a Jump-ReLU activation function, which we define here with a threshold of 1 without loss of generality.
We also analytically estimate sparsity.
