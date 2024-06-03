import torch
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util import np2torch, device
from base_market import MarketMaker

""" use a stochastic policy and run PPO """
class PPO(MarketMaker):
    def __init__(self, input_dim, output_dim, hidden_dim=64, lr=0.001, gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.device = device
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)
        
    def build_actor(self):
        actor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Tanh()
        )
        return actor.to(self.device)
    
    def build_critic(self):
        critic = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        return critic.to(self.device)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_mean = self.actor(state)
        dist = torch.distributions.Normal(action_mean, torch.exp(torch.tensor(0.5)))
        action = dist.sample()
        return action.cpu().detach().numpy()
    
    def loss(self, observations, actions, advantages):
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)
        self.optimizer.zero_grad()
        loss = -torch.mean(log_probs*advantages)
        loss.backward()
        self.optimizer.step()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = self.compute_advantages(td_errors)
        
        # Update actor and critic
        for _ in range(10):
            action_mean = self.actor(states)
            dist = torch.distributions.Normal(action_mean, torch.exp(torch.tensor(0.5)))
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, rewards + self.gamma * next_values * (1 - dones))
            
            entropy_loss = dist.entropy().mean()
            
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def compute_advantages(self, td_errors):
        advantages = np.zeros_like(td_errors)
        advantage = 0
        for t in reversed(range(len(td_errors))):
            advantage = td_errors[t] + self.gamma * advantage * (1 - dones[t])
            advantages[t] = advantage
        return torch.FloatTensor(advantages).to(self.device)