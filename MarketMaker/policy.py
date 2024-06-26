import os
try:
    from MarketMaker.util import np, torch, np2torch, torch2np, build_mlp, normalize, device
    from MarketMaker.rewards import Market
    from MarketMaker.config import Config
except ModuleNotFoundError:
    from util import np, torch, np2torch, torch2np, build_mlp, normalize, device
    from rewards import Market
    from config import Config
import torch.nn as nn
import torch.distributions as ptd
import torch.masked as masked
from tqdm import tqdm
from collections import OrderedDict
import sys

""" Policy module that works with regular or masked tensors/arrays"""

class BasePolicy:
    def action_distribution(self, observations: torch.Tensor):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution
        """
        raise NotImplementedError
    
    def log_probs(self, distribution: ptd.Distribution, actions: torch.Tensor | masked.MaskedTensor):
        """ Return log probs of actions """
        if isinstance(actions, masked.MaskedTensor):
            data = torch.nan_to_num(actions._masked_data, 1).to(device)
            mask = actions._masked_mask[...,0].detach()  # bc probs are 1D
            req_grad = actions.requires_grad
            probs = distribution.log_prob(data).detach()
            return masked.masked_tensor(probs, mask, requires_grad=req_grad).to(device)
        return distribution.log_prob(actions).to(device)
    
    def entropy(self, distribution: ptd.Distribution, observations: torch.Tensor | masked.MaskedTensor):
        """ Return entropy of the distribution """
        entropy = distribution.entropy()
        if not isinstance(observations, masked.MaskedTensor):
            return entropy     # bc entropy is 1D
        mask = observations._masked_mask[...,0].detach()
        req_grad = observations.requires_grad
        return masked.masked_tensor(entropy.detach(), mask, requires_grad=req_grad).to(device)

    def act(self, observations: np.ndarray, return_log_prob = False):
        """ Return np.ndarray actions (and log probs)? from action distribution """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        actions = distribution.sample()
        sampled_actions = torch2np(actions)
        if return_log_prob:
            log_probs = torch2np(self.log_probs(distribution, actions))
            return sampled_actions, log_probs
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    """ categorical policy network for a discrete action space """
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """ Discrete action distribution """
        vals = self.network(observations)
        if isinstance(vals, masked.MaskedTensor):
            vals = vals._masked_data.nan_to_num(0)
        return ptd.Categorical(logits=vals)


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        nn.Module.__init__(self)
        self.network = network
        self.log_std = nn.Parameter(np2torch(np.zeros(action_dim)), requires_grad=False)

    def std(self):
        """ Returns: torch.Tensor of shape [dim(action space)] """
        return torch.exp(self.log_std)

    def action_distribution(self, observations):
        """ continuous action distribution for given observations """
        means = self.network(observations)
        if isinstance(means, masked.MaskedTensor):
            means = means._masked_data.nan_to_num(0)
        return ptd.MultivariateNormal(means, scale_tril=torch.diag(torch.exp(self.log_std)))

##########################################

class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, config: Config):
        super().__init__()
        self.network = build_mlp(config.val_dim, 1, config.n_layers, config.layer_size)
        self.optimizer = torch.optim.Adam(params=self.network.parameters(),lr=config.lr)
    
    def forward(self, observations: torch.Tensor):
        values = self.network(observations)
        if isinstance(values, masked.MaskedTensor):
            """ implement a masked tensor squeeze """
            mask = values._masked_mask.squeeze()
            data = values._masked_data.squeeze()
            req_grad = values.requires_grad
            return masked.masked_tensor(data, mask, requires_grad=req_grad).to(device)
        return values.squeeze()

    def calculate_advantage(self, returns: np.ndarray, observations: np.ndarray):
        """ Compute advantages given returns """
        return returns - torch2np(self.forward(np2torch(observations)))

    def update_baseline(self, returns: np.ndarray, observations: np.ndarray):
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
    def __init__(self, config: Config, market = None | Market) -> None:
        """
        Initialize Policy Gradient Class
            - config: class with all parameters
        """
        self.config = config
        self.discount = config.discount
        # set the baseline (value) network (val_dim -> 1)
        self.baseline = BaselineNetwork(config) if config.use_baseline else None
        self.eps_clip = self.config.eps_clip
        self.do_clip = self.config.do_clip
        self.entropy_coef = self.config.entropy_coef
        self.do_ppo = self.config.do_ppo
        self.lambd = self.config.lambd
        # option to intialize policy on startup? why not
        if market:
            self.init_policy(market)

    def start_dict(self):
        """ map obs_dim straight to act_dim linearly 
            used to initialize network configurations that 
            have never been seen before """
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

    def init_policy(self, market: Market, ne=1000,nb=100, new_train=False):
        """ Initialize self.policy network to match intial market 
        if new_train, will also match policy to the initial state of the market """
        network = build_mlp(self.config.obs_dim, self.config.act_dim, self.config.n_layers, self.config.layer_size)
        self.policy = CategoricalPolicy(network) if self.config.discrete else GaussianPolicy(network, self.config.act_dim)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=self.config.lr)
        # load previous policy if it exists
        # actually train the policy to match the initial state of the market
        # so that it learn to place positive n_ask, n_bid first
        
        if os.path.exists(self.config.network_out) or not new_train:
            self.policy.load_state_dict(torch.load(self.config.network_out))
            return f'initialized policy from {self.config.network_out}'
        state_dict = self.start_dict()
        self.policy.load_state_dict(state_dict)
        if not new_train:
            return f'initialized policy with default state_dict to {self.config.network_out}'
        save_after = ne/20
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=0.02)      
        obs_dim = self.config.obs_dim
        act_dim = self.config.act_dim
        n_obs = self.config.n_obs
        with tqdm(total=ne) as pbar:
            pbar.set_description("Initializing Policy")
            for i in range(ne):
                states = []; actions = []
                states = np.empty((nb, obs_dim))
                actions = np.empty((nb, act_dim))
                past_states = tuple()
                for b in range(nb):
                    market.reset()
                    state = market.state()
                    time_left = (self.config.max_t,)
                    if not len(past_states):
                        past_states = (state,) * n_obs + time_left
                    past_states = past_states[4:-1] + state + time_left
                    states[b] = past_states
                    #action = market.act(state)
                    actions[b] = state
                self.match(states, actions)
                if (i+1) % save_after == 0:
                    if os.path.exists(self.config.network_out):
                        os.remove(self.config.network_out)
                    torch.save(self.policy.state_dict(), self.config.network_out)
                pbar.update(1)
        if os.path.exists(self.config.network_out):
            os.remove(self.config.network_out)
        torch.save(self.policy.state_dict(), self.config.network_out)
        self.optimizer = torch.optim.Adam(params=self.policy.parameters(),lr=self.config.lr)
        return f'saved policy network to {self.config.network_out}'

    def get_returns(self, rewards: np.ndarray | np.ma.MaskedArray) -> np.ndarray | np.ma.MaskedArray:
        """ Classic returns from batched rewards of shape (nb x nt) """
        isMasked = isinstance(rewards, np.ma.MaskedArray)
        returns = np.empty_like(rewards)
        if isMasked:
            mask = rewards.mask
            rewards = rewards.filled(0)   # we want to add when OR on the masks
        returns[:, -1] = rewards[:, -1]
        for t in reversed(range(rewards.shape[1]-1)):
            returns[:, t] = rewards[:, t] + self.discount*rewards[:, t+1]
        if isMasked:
            returns = np.ma.masked_array(returns, mask)
        return returns
    
    def get_td_returns(self, rewards: np.ndarray | np.ma.MaskedArray, observations: np.ndarray | np.ma.MaskedArray) -> np.ndarray | np.ma.MaskedArray:
        """ Compute TD(λ) returns """
        isMasked = isinstance(rewards, np.ma.MaskedArray)
        returns = np.empty_like(rewards)
        values = torch2np(self.baseline.forward(np2torch(observations)))
        if isMasked:
            mask = rewards.mask
            rewards = rewards.filled(0)  # we want to add when OR on the masks
            values = values.filled(0)
        returns[:, -1] = rewards[:, -1] + self.discount * values[:, -1]
        for t in reversed(range(rewards.shape[1]-1)):
            returns[:, t] = rewards[:, t] + self.discount * ((1 - self.lambd) * values[:, t+1] + self.lambd * returns[:, t+1])
        if isMasked:
            returns = np.ma.masked_array(returns, mask)
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

    def get_uneven_td_returns(self, paths: list) -> np.ndarray:
        """ Comupte uneven TD(λ) returns """
        all_returns = []
        finals = []
        for path in paths:
            rewards = path['rew']
            returns = np.empty_like(rewards)
            values = torch2np(self.baseline.forward(np2torch(path['tra'])))
            returns[-1] = rewards[-1]
            G = rewards[-1] + self.config.discount * values[-1]
            returns[-1] = G
            finals.append(G)
            for t in reversed(range(len(rewards)-1)):
                returns[t] = rewards[t] + self.config.discount * ((1 - self.config.lambd) * values[t+1] + self.config.lambd * returns[t+1])
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
        log_probs = self.policy.log_probs(distribution, actions)
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


class Policy(PolicyGradient):
    def __init__(self, config: Config):
        super().__init__(config)
        # only difference is using the right update policy function
        if self.do_ppo:
            self.update_policy = self.update_policy_ppo

    def update_policy_ppo(self, observations, actions, advantages, old_logprobs):
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
        log_probs    = self.policy.log_probs(distribution, actions)
        z_ratio      = torch.exp(log_probs - old_logprobs)
        entropy_loss = self.policy.entropy(distribution, observations)
        if self.do_clip:
            clip_z   = torch.clip(z_ratio,1-self.eps_clip,1+self.eps_clip)
            minimum  = torch.min(z_ratio*advantages,clip_z*advantages)
        else:
            minimum  = z_ratio * advantages
        self.optimizer.zero_grad()
        loss         = -torch.mean(minimum) - torch.mean(self.entropy_coef * entropy_loss)
        loss.backward()
        self.optimizer.step()