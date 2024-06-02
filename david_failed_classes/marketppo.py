import numpy as np
import torch
import torch.nn.functional as F
import argparse, os
from general import get_logger, Progbar, export_plot
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient
from limit_order_book_old import OrderBook

class PPO(PolicyGradient):
    def __init__(self, env, config, seed, logger=None):
        config.use_baseline = True
        super(PPO, self).__init__(env, config, seed, logger)
        self.eps_clip = self.config.eps_clip

    def update_policy(self, observations, actions, advantages, old_logprobs):
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)
        z_ratio = torch.exp(log_probs - old_logprobs)
        clip_z = torch.clip(z_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        minimum = torch.min(z_ratio * advantages, clip_z * advantages)
        self.optimizer.zero_grad()
        loss = -torch.mean(minimum)
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
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, old_logprobs)

            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def sample_path(self, env, num_episodes=None):
        episode = 0
        episode_rewards = [] 
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob=True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                old_logprobs.append(old_logprob)
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
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards


class MarketMakerEnv:
    def __init__(self, inventory, wealth, dt=1e-3, 
                 gamma=1, sigma=1e-2, terminal_time=1,
                 discount=0.99):
        self.I = inventory
        self.W = wealth
        self.book = OrderBook()
        # --- market order parameters --- #
        # assume environment only does market orders
        # rate of market orders
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 20  # average "volume" of market orders at each step
        self.Lambda_s = 20  
        self.K_b = 1  # as K increases, rate w delta decreases
        self.K_s = 1
        # action stuff
        self.sigma = sigma
        self.gamma = gamma 
        self.terminal_time = terminal_time  # second
        self.dt = dt   # millisecond
        # reward stuff
        self.a = 1  # how much we weigh dW
        self.b = 1  # how much we weigh dI
        self.discount = discount

    def reset(self):
        self.initialize_book()
        self.state = self.observe_state()
        return self.state
    
    def initialize_book(self, mid=100, spread=10, nstocks=100, nsteps=100, substeps=1):
        """ Randomly initialize order book """
        if self.book: del(self.book)
        self.book = OrderBook()
        self.book.bid(nstocks//2, mid-spread/2)
        self.book.ask(nstocks//2, mid+spread/2)
        self.midprice = mid
        for t in range(nsteps):
            # perform random MARKET ORDERS
            dW, dI = self.market_step(substeps)
            state  = self.observe_state()
            # perform random LIMIT ORDERS
            self.act(state)

    # --- ENVIRONMENT / DYNAMICS --- #
    def lambda_buy(self, delta_a):
        k = self.alpha * self.K_b
        A = self.Lambda_b / self.alpha
        return A * np.exp(-k*delta_a)

    def lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * np.exp(-k*delta_b)
    
    def market_step(self, nsteps=1):
        """ Evolve market order book with market orders 
        - returns change in wealth, inventory (dW, dI) """
        dW = dI = 0
        for step in range(nsteps):
            delta_b = self.book.delta_b; delta_a = self.book.delta_a
            nbuy  = np.random.poisson(self.lambda_buy(delta_a))
            nsell = np.random.poisson(self.lambda_sell(delta_b))
            n_ask_lift, bought = self.book.buy(nbuy)
            n_bid_hit, sold = self.book.sell(nsell)
            dW += bought - sold; dI += n_bid_hit - n_ask_lift
        return dW, dI
    
    def limit_act(self, n_bid, bid_price, n_ask, ask_price):
        """ Perform a limit order action, making (up to) 2 limit orders: a bid and ask
        - if n_bid or n_ask < 0.5, will not bid or ask """
        # make sure that bid price is always less than ask price
        bid_price = np.clip(bid_price, 0, min(bid_price, self.book.low_ask))
        ask_price = np.clip(ask_price, max(self.book.high_bid, ask_price), np.inf)
        # can only bid integer multiples
        n_bid = round(n_bid)
        n_ask = round(n_ask)
        # LOB automatically only inserts price in cents so dw ab
        # bid_price = round(bid_price.item(), 2)
        self.book.bid(n_bid, bid_price)
        self.book.ask(n_ask, ask_price)
        return n_bid, bid_price, n_ask, ask_price

    def step(self, action):
        n_bid, bid_price, n_ask, ask_price = action
        self.limit_act(n_bid, bid_price, n_ask, ask_price)
        dW, dI = self.market_step()
        self.state = self.observe_state()
        reward = self.calculate_reward(dW, dI)
        done = self.check_done()
        return self.state, reward, done, {}

    def observe_state(self):
        return self.book.nhigh_bid, self.book.high_bid, self.book.nlow_ask, self.book.low_ask

    def calculate_reward(self, dW, dI):
        return dW + dI * self.book.midprice

    def check_done(self):
        return False  # Define your termination condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "-n_epochs", dest='ne', type=int, default=100)
    parser.add_argument("-nb", "-n_batches", dest='nb', type=int, default=1000)
    parser.add_argument("-nt", "-n_times", dest='nt', type=int, default=10000)
    parser.add_argument("-test_initialization", dest='testinitial', default=False, action='store_true')
    parser.add_argument("-l", "--load", nargs='+', default=[])

    args = parser.parse_args()

    config = {
        "output_path": "./outputs/",
        "log_path": "./logs/",
        "record_path": "./records/",
        "plot_output": "./plots/",
        "scores_output": "./scores/",
        "env_name": "MarketMakerEnv",
        "batch_size": 1000,
        "num_batches": 100,
        "max_ep_len": 1000,
        "update_freq": 10,
        "summary_freq": 10,
        "record_freq": 10,
        "eps_clip": 0.2,
        "use_baseline": True,
        "normalize_advantage": True,
    }

    env = MarketMakerEnv(config)
    ppo = PPO(env, config, seed=42)

    if args.testinitial:
        while input("Continue? (y/n): ").strip().lower() == 'y':
            env.reset()
            env.book.plot()
    elif len(args.load) >= 3:
        ppo.load(*args.load)
        ppo.train()
    else:
        ppo.train()

