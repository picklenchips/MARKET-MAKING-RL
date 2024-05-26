from limit_order_book import OrderBook, LimitOrder
from util import uFormat, mpl, plt, np, np2torch
import torch
import torch.nn as nn
import tqdm as tqdm
from scipy import stats

# v(x,s,q,t), 
# x = initial wealth, s = initial stock value, q = nstocks, t = time
def frozen_value(initial_wealth, stock_val, nstocks, time):
    first = -np.exp(-gamma*(initial_wealth+nstocks*stock_val))
    second = np.exp((gamma*nstocks*sigma)**2 * (T - time) / 2)
    return first * second

def res_ask_price(s,q,t):
    return s + (1-2*q) * gamma * sigma**2 * (T-t)

def res_bid_price(s,q,t):
    return s - (1+2*q) * gamma * sigma**2 * (T-t)

# avg between bid and ask
def res_price(s, q, t_left):  # reservation / indifference price
    return s - q * gamma * sigma**2 * t_left

"""
model trading intensity.
assume constant frequency f^Q(x)\alpha x^{-1-\alpha}
"""
class Exponential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return -abs(self.a) * torch.exp(-abs(self.b) * x)

    def string(self):
        return f'y = -{abs(self.a.item())} * exp(-{abs(self.b.item())} x)'



class MarketMaker():
    def __init__(self, inventory, wealth, dt=1e-3, 
                 gamma=1, sigma=1e-2, terminal_time=1):
        self.I = inventory
        self.W = wealth
        self.book = OrderBook()
        # --- market order parameters --- #
        # assume environment only does market orders
        # rate of market orders
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 30
        self.Lambda_s = 30
        self.K_b = 1
        self.K_s = 1
        # action stuff
        self.sigma = sigma
        self.gamma = gamma  # discount variable
        self.terminal_time = terminal_time  # second
        self.dt = dt   # millisecond
        # NN dimensions
        self.obs_dim = 4
        self.act_dim = 4
        self.val_dim = 8
    
    def initialize_networks(self, obs_dim=4, act_dim=4, value_dim=8, hidden_dim=10):
        """ set policy and value networks 
        - value_dim can change depending on if we track (obs, rew) or 
        - (obs, act, rew) or (obs, obs1, rew) or (obs, act, obs1, rew) or ..."""
        #obs_dim = 4  # (n_bid, bid_price, n_ask, ask_price)
        #act_dim = 4  # (n_bid, bid_price, n_ask, ask_price)
        #rew_dim = 4  # (wealth[t-1], inventory[t-1], midprices[t-1], time_left)
        #hidden_dim = 10
        # POLICY NETWORK
        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(hidden_dim, act_dim))
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        # VALUE NETWORK
        self.value = nn.Sequential(nn.Linear(value_dim, hidden_dim), 
                                           nn.LeakyReLU(),
                                           nn.Linear(hidden_dim, 1),
                                           Exponential())
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=1e-3)
    
    # --- trading intensities / environment --- #
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
            action = self.act(state, naive=2)

    # --- RL part --- #
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
            action = self.policy(torch.tensor(state)).item()
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

    def immediate_reward(reward_state, a=1, b=1):
        """ immediate + discounted future reward """
        dW, dI, time_left = reward_state
        return a*dW + np.exp(-b*time_left) * np.sign(dI)
    
    def final_reward(wealth, inventory, midprice):
        return -np.exp(-(wealth + inventory*midprice))

    def reservation_price(self, midprice, inventory, t_left):  # reservation / indifference price
        return midprice - inventory * self.gamma * self.sigma**2 * t_left

    def optimal_spread(self, time_left):
        return self.gamma * self.sigma**2 * time_left + 2*np.log(1+2*self.gamma/(self.alpha*(self.K_b+self.K_a)))/self.gamma
    
    def simulate(self, num = 1000, track_all=False, action=''):
        """ naively iterate through order book to show evolution over time using current policy algorithm 
        Sample a batch of trajectories under some policy
        Inputs:
        - num = number of trajectories to sample
        - track_all = keep track of wealth, inventory, midprices over time
        - action = 'naive' or 'avellaneda', or anything else
        Output: (trajectories, rewards)
        - trajectories = (num x num_times x val_dim) nd.arary
        - rewards = (num x num_times) nd.array
        if track_all, output: (trajectories, rewards, wealth, inventory, midprices) """
        T = self.terminal_time; dt = self.dt
        nt = T // dt
        trajectories = np.empty((num, nt, self.val_dim))
        rewards = np.empty((num, nt))
        if track_all:  # track all for later plotting?
            wealth = np.empty((num, nt))
            inventory = np.empty((num, nt))
            midprices = np.empty((num, nt))
        with tqdm(total=num) as pbar:
            pbar.set_description("Creating Batch...")
            for b in range(num):
                self.initialize_book()
                reward_states = []
                W = self.W
                I = self.I
                if track_all:
                    wealth[b, 0] = self.W
                    inventory[b, 0] = self.I
                    midprices[b, 0] = self.book.midprice
                for t in range(nt):
                    time_left = (self.teminal_time - t*self.dt,)
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
                    reward_states.append(reward_state)
                    trajectories[b, t] = state + action + reward_state
                # --- GET REWARDS from reward_state --- #
                rewards[b, -1] = self.final_reward(wealth[b, -1], inventory[b, -1], midprices[b, -1])
                for i in range(nt-2, -1, -1):
                    rewards[b, i] = self.immediate_reward(reward_states[i]) + mm.gamma*rewards[b, i+1]
                pbar.update(1)
                pbar.set_postfix_str(f"Reward {rewards[b, -1]}", refresh=True)
        if track_all:
            return trajectories, rewards, wealth, inventory, midprices
        return trajectories, rewards
    
    def plot(self, wealth, inventory, midprices, title=''):
        """ plot hshit """
        times = np.arange(0, self.terminal_time, self.dt)
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        for i, y, name in zip((0,1,2),[wealth, inventory, midprices],'Wealth', 'Inventory', 'Midprice'):
            ax = axs[i]
            ax.plot(times, y)
            ax.set(xlabel="Time")
            ys = np.mean(y, axis=0)
            yerrs = stats.sem(y, axis=0)
            ax.fill_between(times, ys - yerrs, ys + yerrs, alpha=0.25)
            ax.plot(times, ys, label=name)
        axs[2].set(xlabel="Time")
        axs[0].plot(wealth + inventory*midprices,label='Total Value')
        axs[0].legend()
        if title: plt.title(title)
        plt.show()
    
    def get_advantages(self, rewards, trajectories):
        """ Compute advantages from rewards and trajectories 
        - 1. run value network on trajecotires """
        advantages = np.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards))):
            advantage = rewards[t] + self.gamma * advantage
            advantages[t] = advantage
        return advantages


    def train_value(self):
        # -- TRAIN REWARD FUNCTION -- #
        new_val  = np2torch(reward)
        pred_val = self.value(value_state)
        loss = -torch.mean((new_val - pred_val)**2)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()


# ---- MONTE CARLO TRAIN THIS ---- #

# tuple of time, wealth_diff, inventory_diff, midprice_diff
# from https://discovery.ucl.ac.uk/id/eprint/10116730/1/RLforHFMM.pdf 


# b = gamma in Trainer class?

def train_market():
    """ Monte-Carlo Ish Thing """
    obs_dim = act_dim = 4
    rew_dim = 3
    # for value function, we need 
    # (observation, action, time_left, and (wealth, inventory, midprice) at final time)
    # as we are using MC value function
    value_dim = obs_dim + act_dim + rew_dim

    # initialize parameters
    dt = 0.005
    nt = 10000  # define horizon
    terminal_time = nt*dt  # will never exactly reach this in for loop
    times = np.arange(0, terminal_time, dt)
    mm = MarketMaker(0, 0, dt=dt, gamma=1, sigma=1, terminal_time=terminal_time)

    #trajectores[:obs_dim], [obs_dim:act_dim], [act_dim:] for observations, actions, values
    trajectories, rewards, wealth, inventory, midprice = mm.simulate(mm, batch_size = 1000, track_all=True)
    mm.plot(wealth, inventory, midprice, title='1000 batches')
    # (batch_size x nt x value_dim), (batch_size x nt)
    # initialize policy function in self.policy and value in self.value
    mm.initialize_networks(value_dim=value_dim)
    # backpropagate through trajectory to define reward from t?
    loss = -(rewards - mm.value(trajectories)).mean()
    for i in range(len(trajectory)):
        state, action, reward = trajectory[i]
        value_state = state + action + (final_time - i*dt, (tot_wealth, tot_inventory, mm.book.midprice))
        new_val = immediate_reward(reward)
        pred_val = mm.value(value_state)
        loss = -(new_val - pred_val)
        mm.value_optimizer.zero_grad()
        loss.backward()
        mm.value_optimizer.step()

        # update policy
        advantage = advantages[i] - mm.value(trajectory)
        normalized_advantage = (advantage - np.mean(advantage))/(np.std(advantage)+1e-10)
        # compute discounted advantage using discounted returns?
        # backstep
        mm.policy_optimizer.zero_grad()
        loss.backward()
        mm.policy_optimizer.step()
    # use advantage function to debias 
    advantage = trajectory_return - mm.value(trajectory)
    normalized_advantage = (advantage - np.mean(advantage))/(np.std(advantage)+1e-10)
    # compute discounted advantage using discounted returns?
    # backstep
    

    # update progress bar every episode
    pbar.update(1)
    pbar.set_description("text", refresh=True)
    pbar.set_postfix_str(f"Reward {trajectory_return}")

    discount = 0.99
            
            # --- UPDATE POLICY AND VALUE FUNCTIONS --- #




        
        

        

        











if __name__ == "__main__":
    mm = MarketMaker(0, 0)
    mm.step()
