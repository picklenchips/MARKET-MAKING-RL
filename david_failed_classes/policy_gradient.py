import numpy as np
import torch
import os
import gym
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy

class PolicyGradient:
    def __init__(self, env, config, seed, logger=None):
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.seed = seed

        self.logger = logger or get_logger(config.log_path)
        self.env = env
        torch.manual_seed(self.seed)

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]

        self.lr = self.config.learning_rate
        self.init_policy()

        if config.use_baseline:
            self.baseline_network = BaselineNetwork(env, config)

    def init_policy(self):
        network = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        self.policy = CategoricalPolicy(network) if self.discrete else GaussianPolicy(network, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
        if scores_eval:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env, num_episodes=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.policy.act(states[-1][None])[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            T = len(rewards)
            returns = [0] * T
            returns[T-1] = rewards[T-1]
            for i in range(T-1):
                idx = T - i - 2
                returns[idx] = rewards[idx] + self.config.gamma * returns[idx + 1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def normalize_advantage(self, advantages):
        return (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

    def calculate_advantage(self, returns, observations):
        if self.config.use_baseline:
            advantages = self.baseline_network.calculate_advantage(returns, observations)
        else:
            advantages = returns
        if self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)
        return advantages

    def update_policy(self, observations, actions, advantages):
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)
        self.optimizer.zero_grad()
        loss = -torch.mean(log_probs * advantages)
        loss.backward()
        self.optimizer.step()

    def train(self):
        last_record = 0
        self.init_averages()
        all_total_rewards = []
        averaged_total_rewards = []

        for t in range(self.config.num_batches):
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(t, avg_reward, sigma_reward)
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(averaged_total_rewards, "Score", self.config.env_name, self.config.plot_output)

    def evaluate(self, env=None, num_episodes=1):
        if env is None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.evaluate(env, 1)

    def run(self):
        if self.config.record:
            self.record()
        self.train()
        if self.config.record:
            self.record()
