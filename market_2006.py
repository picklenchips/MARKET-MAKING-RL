from limit_order_book import OrderBook
from util import uFormat, mpl, plt, np, np2torch
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats
import argparse

class MarketMaker():
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
        self.Lambda_b = 1
        self.Lambda_s = 1
        self.K_b = 1
        self.K_s = 1
        # action stuff
        self.sigma = sigma
        self.gamma = gamma 
        self.terminal_time = terminal_time  # second
        self.dt = dt   # millisecond
        # NN dimensions
        self.obs_dim = 4
        self.act_dim = 4
        self.val_dim = 8
        # reward stuff
        self.a = 1  # how much we weigh dW
        self.b = 1  # how much we weigh dI
        self.discount = discount
    
    def initialize_networks(self, obs_dim=5, act_dim=4, value_dim=11, hidden_dim=10):
        """ set policy and value networks 
        - value_dim can change depending on if we track (obs, rew) or 
        - (obs, act, rew) or (obs, obs1, rew) or (obs, act, obs1, rew) or ..."""
        #obs_dim = 4  # (n_bid, bid_price, n_ask, ask_price)
        #act_dim = 4  # (n_bid, bid_price, n_ask, ask_price)
        #rew_dim = 4  # (wealth[t-1], inventory[t-1], midprices[t-1], time_left)
        #hidden_dim = 10
        # POLICY NETWORK
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.val_dim = value_dim
        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LeakyReLU(),
                                    nn.Linear(hidden_dim, act_dim))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        # VALUE NETWORK
        self.value = nn.Sequential(nn.Linear(value_dim, hidden_dim), 
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(),
                                   nn.Linear(hidden_dim, 1))
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-3)
    
# --- ENVIRONMENT / DYNAMICS --- #
    def lambda_buy(self, delta_a):
        k = self.alpha * self.K_b                                                                                                                                                                                                                                                      
        A = self.Lambda_b / self.alpha
        return A * np.exp(-k*delta_a)

    def lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * np.exp(-k*delta_b)
    
    def market_step(self):
        """ Evolve market order book with market orders 
        - returns change in wealth, inventory (dW, dI) """
        delta_b = self.book.delta_b; delta_a = self.book.delta_a
        nbuy  = np.random.poisson(self.lambda_buy(delta_a))
        nsell = np.random.poisson(self.lambda_sell(delta_b))
        n_ask_lift, bought = self.book.buy(nbuy)
        n_bid_hit, sold = self.book.sell(nsell)
        return bought - sold, n_bid_hit - n_ask_lift

    def initialize_book(self, initial_midprice=100, initial_spread=10, initial_num_stocks=100, 
                        nsteps=30):
        """ Randomly initialize order book """
        if self.book: del(self.book)
        self.book = OrderBook()
        self.book.bid(initial_num_stocks//2, initial_midprice-initial_spread/2)
        self.book.ask(initial_num_stocks-initial_num_stocks//2, initial_midprice+initial_spread/2)
        # keep track
        for t in range(nsteps):
            # perform random MARKET ORDERS
            dW, dI = self.market_step()
            # can keep track of wealth, inventory changes maybe
            # wealth += bought - sold
            state  = self.observe_state()
            # perform random LIMIT ORDERS
            action = self.act(state)

# --- STATES / ACTIONS --- #
    def observe_state(self):
        """ instead of midprice, delta_a, delta_b, can just use actual price of the 
        highest bid and lowest ask (dim=4) """
        return self.book.nhigh_bid, self.book.high_bid, self.book.nlow_ask, self.book.low_ask

    def act(self, state: tuple):
        """ Perform action on order book (dim=4)
        NAIVE = 0 (default):
        - Inputs: (n_bid, bid_price, n_ask, ask_price, time_left)
        Inputs: (n_bid, bid_price, n_ask, ask_price, time_left)
        - action is only limit orders (for now) 
            - can be extended to (n_bid, bid_price, n_ask, ask_price, n_buy, n_sell)
        - returns action taken, (n_bid, bid_price, n_ask, ask_price)
        - naive sets price variance of 'stupid' policy taken around midprice
        """
        try:  # invalid input action??
            if len(state) < 3: raise TypeError
        except TypeError:  # resample state
            state = (self.book.nhigh_bid, self.book.high_bid, self.book.nlow_ask, self.book.low_ask)
        if len(state) == 5:  # use NN policy
            n_bid, bid_price, n_ask, ask_price, time_left = state
            action = self.policy(np2torch(np.array(state))).detach().cpu().numpy()
        if len(state) == 4:  # default "naive" policy
            delta_b = self.book.delta_b; delta_a = self.book.delta_a
            n_bid, bid_price, n_ask, ask_price = state
            bid_price = np.random.normal(bid_price, delta_b/4)
            ask_price = np.random.normal(ask_price, delta_a/4)
            n_bid = np.random.poisson(n_bid)
            n_ask = np.random.poisson(n_ask) 
            action = (n_bid, bid_price, n_ask, ask_price)
        if len(state) == 3:  # avellaneda policy
            wealth, inventory, time_left = state
            n_bid = n_ask = 1  #TODO i think...
            # greedily try to set midprice to be reservation price
            midprice = self.book.midprice
            res_price = self.reservation_price(midprice, inventory, time_left)
            optimal_spread = self.optimal_spread(time_left)
            # quote around this...
            bid_price = res_price - optimal_spread / 2
            ask_price = res_price + optimal_spread / 2
            n_bid = np.random.poisson(n_bid)
            n_ask = np.random.poisson(n_ask)
        self.limit_act(n_bid, bid_price, n_ask, ask_price)
        return (n_bid, bid_price, n_ask, ask_price)
    
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

# --- REWARD FUNCTIONS --- #
    # objective with no inventory dynamics?
    def frozen_reward(self, initial_wealth, stock_val, nstocks, time_left):
        first = -np.exp(-self.gamma*(initial_wealth + nstocks*stock_val))
        second = np.exp((self.gamma*nstocks*self.sigma)**2 * time_left / 2)
        return first * second

    def immediate_reward(self, r_state):
        """ immediate + discounted future reward """
        # dW, dI, time_left = reward_state
        if isinstance(r_state, tuple):
            r_state = np.array(r_state)
        return self.a*r_state[...,0] + np.exp(-self.b*r_state[...,2]) * np.sign(r_state[...,1])
    
    def final_reward(self, dW, inventory, midprice):
        return dW + inventory*midprice

    def reservation_price(self, midprice, inventory, t_left):  # reservation / indifference price
        return midprice - inventory * self.gamma * self.sigma**2 * t_left

    def optimal_spread(self, time_left):
        return self.gamma * self.sigma**2 * time_left + 2*np.log(1+2*self.gamma/(self.alpha*(self.K_b+self.K_a)))/self.gamma
    
# --- SIMULATION --- #
    def simulate(self, nbatch = 1000, track_all=False, action=''):
        """ naively iterate through order book to show evolution over time using current policy algorithm 
        Sample a batch of trajectories under some policy
        Inputs:
        - nbatch = number of trajectories to sample
        - track_all = keep track of wealth, inventory, midprices over time
        - action = 'naive' or 'avellaneda', or anything else
        Output: (trajectories, rewards)
        - trajectories = (nbatch x num_times x val_dim) nd.arary
        - rewards = (nbatch x num_times) nd.array
        if track_all, output: (trajectories, rewards, wealth, inventory, midprices) """
        T = self.terminal_time; dt = self.dt
        nt = int(T/dt)
        trajectories = np.empty((nbatch, nt, self.val_dim))
        rewards = np.empty((nbatch, nt))
        if track_all:  # track all for later plotting?
            wealth = np.empty((nbatch, nt))
            inventory = np.empty((nbatch, nt),dtype=int)
            midprices = np.empty((nbatch, nt))
        with tqdm(total=nbatch) as pbar:
            pbar.set_description("Creating Batch...")
            for b in range(nbatch):
                self.initialize_book()
                W = self.W; I = self.I
                if track_all:
                    wealth[b, 0] = self.W
                    inventory[b, 0] = self.I
                    midprices[b, 0] = self.book.midprice
                for t in range(nt):
                    time_left = (self.terminal_time - t*self.dt,)
                    state = self.observe_state()
                    # AVELLANEDA ACTION
                    if action == 'avellaneda':
                        action = self.act((W, I)+time_left)
                    elif action == 'naive':
                        action = self.act(state)
                    else:  # use NN policy
                        action = self.act(state+time_left)
                    dW, dI = self.market_step()
                    # OBSERVE / STORE
                    W += dW; I += dI
                    if track_all:
                        wealth[b, t] = wealth[b, t-1] + dW
                        inventory[b, t] = inventory[b, t-1] + dI
                        midprices[b, t] = self.book.midprice
                    #TODO: do we want to reward based on intermediate actual 
                    # wealth, inventory instead of just dW, dI???
                    reward_state = (dW, dI) + time_left
                    rewards[b, t] = self.immediate_reward(reward_state)
                    trajectories[b, t] = state + action + reward_state
                # --- GET REWARDS from reward_state --- #
                rewards[b, t] = self.final_reward(dW, inventory[b, -1], midprices[b, -1])
                pbar.update(1)
                pbar.set_postfix_str(f"Reward {rewards[b, -1]}", refresh=True)
        if track_all:
            return trajectories, rewards, wealth, inventory, midprices
        return trajectories, rewards

    def get_returns(self, rewards):
        """ Compute returns from batched rewards and batched trajectories using discounted sum
        Inputs:
            - rewards (nbatch x nt) np.ndarray
            - trajectories (nbatch x nt x val_dim) np.ndarray """
        if isinstance(rewards, np.ndarray):
            returns = np.empty_like(rewards)
        else:
            returns = torch.empty_like(rewards)
        nt = rewards.shape[1]
        for t in range(nt-2, -1, -1):
            returns[:, t] = rewards[:, t] + self.discount*rewards[:, t+1]
        return returns
    
    def plot(self, wealth, inventory, midprices, title=''):
        """ plot data from a batch of trajectories
        Inputs: (nbatch x nt) np.ndarrays """
        times = np.arange(0, self.terminal_time, self.dt)
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        for i, y, name in zip((0,1,2),(wealth, inventory, midprices),('Wealth', 'Inventory', 'Midprice')):
            ax = axs[i]
            ax.set(ylabel=name)
            ys = np.mean(y, axis=0)
            yerrs = stats.sem(y, axis=0)
            ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{i}")
            ax.plot(times, ys, color=f"C{i}")
        axs[2].set(xlabel="Time")
        i += 1
        y = wealth + inventory*midprices
        ys = np.mean(y, axis=0); yerrs = stats.sem(y, axis=0)
        axs[0].fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25, color=f"C{i}")
        axs[0].plot(times, ys, label='Total Value', color=f"C{i}")
        axs[0].legend()
        if title: plt.title(title)
        plt.show()

# --- TRAINING --- #
    def get_advantages(self, trajectories, returns):
        """ Compute advantages from batched rewards and batched trajectories 
        - returns (nbatch x nt) np.ndarray
        - trajectories (nbatch x nt x val_dim) np.ndarray
        - run value network on trajectories to get value estimates """
        trajectories = np2torch(trajectories)
        advantages = returns - self.value(trajectories).squeeze().detach().cpu().numpy()
        return (advantages - advantages.mean()) / (advantages.std())

    def update_value(self, trajectories, returns):
        """ use MSE loss to train value function 
        Inputs: returns (nbatch x nt) np.ndarray, trajectories (nbatch x nt x val_dim) np.ndarray """
        act_val  = np2torch(returns); trajectories = np2torch(trajectories)
        pred_val = self.value(trajectories).squeeze()
        loss = torch.mean((act_val - pred_val)**2)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def update_policy(self, trajectories, advantages):
        """ use MSE loss to train policy function """
        trajectories = np2torch(trajectories)
        advantages = np2torch(advantages)
        states = trajectories[..., [0,1,2,3,-1]]
        actions = self.policy(states)
        loss = -torch.log(advantages[...,None] * torch.log(actions))
        loss = loss.mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

# ---- MONTE CARLO TRAIN THIS ---- #

# tuple of time, wealth_diff, inventory_diff, midprice_diff
# from https://discovery.ucl.ac.uk/id/eprint/10116730/1/RLforHFMM.pdf 

def train_market(num_epochs = 100, batch_size = 100, timesteps = 1000):
    """ Monte-Carlo Ish Thing """
    obs_dim = act_dim = 4
    rew_dim = 3
    value_dim = obs_dim + act_dim + rew_dim

    # initialize parameters
    dt = 0.005
    nt = timesteps  
    terminal_time = nt*dt
    times = np.arange(0, terminal_time, dt)

    mm = MarketMaker(0, 0, dt=dt, gamma=1, sigma=1, terminal_time=terminal_time)
    mm.initialize_networks(value_dim=value_dim)
    #trajectores[:obs_dim], [obs_dim:act_dim], [act_dim:] for observations, actions, values
    with tqdm(total=num_epochs) as pbar:
        pbar.set_description("Training Market Maker...")
        for epoch in range(num_epochs):
            trajectories, rewards, wealth, inventory, midprice = mm.simulate(nbatch = batch_size, track_all=True)
            mm.plot(wealth, inventory, midprice, title=f'epoch {epoch}')
            returns = mm.get_returns(rewards)
            advantages = mm.get_advantages(trajectories, returns)
            mm.update_value(trajectories, returns)
            mm.update_policy(trajectories, advantages)
            pbar.update(1)
            pbar.set_postfix_str(f"Reward {rewards[:,-1].mean()}", refresh=True)

parser = argparse.ArgumentParser()
parser.add_argument("-ne", "-n_epochs", dest='ne', type=int, default=100)
parser.add_argument("-nb", "-n_batches", dest='nb', type=int, default=100)
parser.add_argument("-nt", "-n_times", dest='nt', type=int, default=10000)

if __name__ == "__main__":
    args = parser.parse_args()
    train_market(args.ne, args.nb, args.nt)