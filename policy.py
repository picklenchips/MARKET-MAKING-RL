import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from config import Config
import logging, os
from util import np2torch, build_mlp, normalize
from base_market import Market
from tqdm import tqdm
from collections import OrderedDict


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution
        """
        raise NotImplementedError

    def act(self, observations, return_log_prob = False):
        """ Return actions (and log probs)? from action distribution """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        actions = distribution.sample()
        sampled_actions = actions.cpu().detach().numpy()
        if return_log_prob:
            log_probs = distribution.log_prob(actions).cpu().detach().numpy()
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    """ categorical policy network for a discrete action space """
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """ Discrete action distribution """
        return ptd.Categorical(logits=self.network(observations))


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=False)

    def std(self):
        """ Returns: torch.Tensor of shape [dim(action space)] """
        return torch.exp(self.log_std)

    def action_distribution(self, observations):
        """ continuous action distribution for given observations """
        return ptd.MultivariateNormal(self.network(observations), scale_tril=torch.diag(self.std()))


##########################################

class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, config):
        super().__init__()
        self.network = build_mlp(config.val_dim, 1, config.n_layers, config.layer_size)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(),lr=config.lr)
    
    def forward(self, observations):
        return self.network(observations).squeeze()

    def calculate_advantage(self, returns, observations):
        """ Compute advantages given returns """
        observations = np2torch(observations)
        return returns - self.forward(observations).cpu().detach().numpy()

    def update_baseline(self, returns, observations):
        """ Gradient baseline to match returns """
        returns = np2torch(returns, True)
        observations = np2torch(observations, True)
        baseline = self.forward(observations)
        self.optimizer.zero_grad()
        loss = torch.mean((baseline - returns)**2)
        loss.backward()
        self.optimizer.step()


##########################################

class PolicyGradient():
    """
    Class for implementing a policy gradient algorithm
    """
    def __init__(self, config: Config) -> None:
        """
        Initialize Policy Gradient Class
            - config: class with all parameters
        """
        self.config = config
        self.discount = config.discount
        # set the baseline (value) network (val_dim -> 1)
        if config.use_baseline:
            self.baseline = BaselineNetwork(config)

    def start_dict(self):
        """ map obs_dim straight to act_dim linearly """
        act_dim = self.config.act_dim
        obs_dim = self.config.obs_dim
        n_layers = self.config.n_layers
        layer_size = self.config.layer_size
        inp = torch.randn((layer_size, obs_dim))/10
        inp.fill_diagonal_(1)
        inp_bias = torch.randn((layer_size))/10
        layers = [('log_std', torch.zeros(act_dim)),('network.0.weight', inp),(f'network.0.bias', inp_bias)]
        for i in range(n_layers-1):
            middle = torch.randn((layer_size, layer_size))/10
            middle.fill_diagonal_(1)
            mid_bias = torch.randn((layer_size))/10
            layers.append((f"network.{2*(i+1)}.weight", middle))
            layers.append((f"network.{2*(i+1)}.bias", mid_bias))
        output = torch.randn((act_dim, layer_size))/10
        output.fill_diagonal_(1)
        out_bias = torch.randn((act_dim))/10
        layers.append((f'network.{2*n_layers}.weight', output))
        layers.append((f'network.{2*n_layers}.bias', out_bias))
        return OrderedDict(layers)

    def init_policy(self, market: Market, ne=1000,nb=100, start_dict=False):
        """ Initialize self.policy network to match intial market """
        network = build_mlp(self.config.obs_dim, self.config.act_dim, self.config.n_layers, self.config.layer_size)
        self.policy = CategoricalPolicy(network) if self.config.discrete else GaussianPolicy(network, self.config.act_dim)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=0.02)
        # load previous policy
        if os.path.exists(self.config.out+"_init-pol.pth"):
            self.policy.load_state_dict(torch.load(self.config.out+"_init-pol.pth"))
            print(f'initialized policy from {self.config.out}_init-pol.pth')
            self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=self.config.lr)
            return
        # INITIALIZE POLICY TO MATCH FIRST OBSERVED STATE
        state_dict = self.start_dict()
        if start_dict:
            self.policy.load_state_dict(state_dict)
        save_after = ne/20
        with tqdm(total=ne) as pbar:
            pbar.set_description("Initializing Policy")
            for i in range(ne):
                states = []; actions = []
                states = np.empty((nb, self.config.obs_dim))
                actions = np.empty((nb, self.config.act_dim))
                for b in range(nb):
                    market.reset()
                    state = market.state()
                    states[b] = state + (self.config.max_t,)
                    #action = market.act(state)
                    actions[b] = state
                self.match(states, actions)
                if (i+1) % save_after == 0:
                    if os.path.exists(self.config.out+"_init-pol.pth"):
                        os.remove(self.config.out+"_init-pol.pth")
                    torch.save(self.policy.state_dict(), self.config.out+"_init-pol.pth")
                pbar.update(1)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=self.config.lr)

    def get_returns(self, rewards: np.ndarray):
        """ Compute discounted returns from batched rewards of shape (nbatch x nt) """
        returns = np.empty_like(rewards)
        returns[:, -1] = rewards[:, -1]
        for t in reversed(range(rewards.shape[1]-1)):
            returns[:, t] = rewards[:, t] + self.discount*rewards[:, t+1]
        return returns
    
    def get_uneven_returns(self, paths: list) -> np.ndarray:
        """ Compute discounted returns from batched rewards of shape (nbatch x nt) """
        all_returns = []
        finals = []
        for path in paths:
            rewards = path['rew']
            returns = np.empty_like(rewards)
            returns[-1] = rewards[-1]
            finals.append(returns[-1])
            for t in reversed(range(len(rewards)-1)):
                returns[t] = rewards[t] + self.discount*rewards[t+1]
            all_returns.append(returns)
        return np.concatenate(all_returns), np.array(finals)
    
    def get_td_returns(self, rewards, values):
        """ Compute TD(λ) returns """
        td_lambda_returns = np.zeros_like(rewards)
        for b in range(rewards.shape[0]):
            G = rewards[b, -1] + self.config.discount * values[b, -1]
            for t in reversed(range(rewards.shape[1]-1)):
                G = rewards[b, t] + self.config.discount * ((1 - self.config.lambd) * values[b, t+1] + self.config.lambd * G)
                td_lambda_returns[b, t] = G
        return td_lambda_returns

    def get_uneven_td_returns(self, paths: list) -> np.ndarray:
        """ Comupte uneven TD(λ) returns """
        all_returns = []
        finals = []
        for path in paths:
            rewards = path['rew']
            returns = np.empty_like(rewards)
            values = self.baseline.forward(np2torch(path['tra'])).cpu().detach().numpy()
            returns[-1] = rewards[-1]
            G = rewards[-1] + self.config.discount * values[-1]
            returns[-1] = G
            finals.append(G)
            for t in reversed(range(len(rewards)-1)):
                G = rewards[t] + self.config.discount * ((1 - self.config.lambd) * values[t+1] + self.config.lambd * G)
                returns[t] = G
            all_returns.append(returns)
        return np.concatenate(all_returns), np.array(finals)
    
    def get_advantages(self, returns, trajectories):
        """ Calculates the advantage for all of the observations
        Inputs:
            - returns (nbatch x nt) np.ndarray
            - trajectories (nbatch x nt x val_dim) np.ndarray
        Output:
            advantages: np.array of shape (nbatch x)
        """
        if self.config.use_baseline:
            advantages = self.baseline.calculate_advantage(returns, trajectories)
        else:
            advantages = returns
        if self.config.normalize_advantages:
            advantages = normalize(advantages)
        return advantages

    def update_policy(self, observations, actions, advantages, old_logprobs=None):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations, True)
        actions = np2torch(actions, True)
        advantages = np2torch(advantages, True)
        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)
        self.optimizer.zero_grad()
        loss = -torch.mean(log_probs*advantages)
        loss.backward()
        self.optimizer.step()

    def match(self, observations, actions):
        """ train actions to match observations """
        observations = np2torch(observations, requires_grad = True)
        actions = np2torch(actions, requires_grad = True)
        distribution = self.policy.action_distribution(observations)
        curr_actions = distribution.sample()
        loss = torch.mean((curr_actions - actions)**2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPO(PolicyGradient):
    def __init__(self, config: Config):
        config.use_baseline = True
        super().__init__(config)
        self.eps_clip = self.config.eps_clip
        self.do_clip = self.config.do_clip
        self.entropy_coef = self.config.entropy_coef

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """ Perform one gradient ascent step on the PPO clipped objective function
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                batch size x nt x act_dim if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]
        """
        observations = np2torch(observations, True)
        actions      = np2torch(actions, True)
        advantages   = np2torch(advantages, True)
        old_logprobs = np2torch(old_logprobs, True)
        
        distribution = self.policy.action_distribution(observations)
        log_probs    = distribution.log_prob(actions)
        z_ratio      = torch.exp(log_probs - old_logprobs)
        entropy_loss = distribution.entropy()
        if self.do_clip:
            clip_z   = torch.clip(z_ratio,1-self.eps_clip,1+self.eps_clip)
            minimum  = torch.min(z_ratio*advantages,clip_z*advantages)
        else:
            minimum  = z_ratio * advantages
        self.optimizer.zero_grad()
        loss         = -torch.mean(minimum) - torch.mean(self.entropy_coef * entropy_loss)
        loss.backward()
        self.optimizer.step()