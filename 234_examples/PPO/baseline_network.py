import numpy as np
import torch
import torch.nn as nn
import gym
from network_utils import build_mlp, device, np2torch


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, etc.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate
        observation_dim = self.env.observation_space.shape[0]

        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############
        self.network = build_mlp(observation_dim, 1, self.config.n_layers, self.config.layer_size)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(),lr=self.lr)
        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #######################################################
        #########   YOUR CODE HERE - 1 lines.     #############
        output = self.network(observations).squeeze()
        #######################################################
        #########          END YOUR CODE.          ############
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############
        advantages = returns - self.forward(observations).cpu().detach().numpy()
        #######################################################
        #########          END YOUR CODE.          ############
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer once.

        You may find the following documentation useful:
        https://pytorch.org/docs/stable/nn.functional.html
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 4-10 lines.  #############
        baseline = self.forward(observations)
        self.optimizer.zero_grad()
        loss = torch.mean((baseline - returns)**2)
        loss.backward()
        self.optimizer.step()
        #######################################################
        #########          END YOUR CODE.          ############
