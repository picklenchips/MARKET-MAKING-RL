import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. Compute the log probability of
        the sampled actions using self.action_distribution. You will have to
        convert the actions and log probabilities to a numpy array, via numpy(). 

        You may find the following documentation helpful:
        https://pytorch.org/docs/stable/distributions.html
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.    ############
        # torch.distributions.Distribution of some type
        # basically your pi_theta
        # .numpy()
        distribution = self.action_distribution(observations)
        actions = distribution.sample()
        sampled_actions = actions.cpu().detach().numpy()
        if return_log_prob:
            log_probs = distribution.log_prob(actions).cpu().detach().numpy()
        #######################################################
        #########          END YOUR CODE.          ############
        if return_log_prob:
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        # compute logits using self.network
        logits = self.network(observations)
        distribution = ptd.Categorical(logits=logits)
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        nn.Module.__init__(self)
        self.network = network
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        #######################################################
        #########          END YOUR CODE.          ############

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions. It can be computed from self.log_std
        """
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        std = torch.exp(self.log_std)
        #######################################################
        #########          END YOUR CODE.          ############
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """
        #######################################################
        #########   YOUR CODE HERE - 2-4 lines.    ############
        means = self.network(observations)
        distribution = ptd.MultivariateNormal(means, scale_tril=torch.diag(self.std()))
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution
