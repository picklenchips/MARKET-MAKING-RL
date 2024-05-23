from limit_order_book import OrderBook, LimitOrder
from util import uFormat, mpl, plt, np
import torch
import torch.nn as nn


"""
ASSUMPTIONS:
1. asssume money market pays no interest
2. mid-price given by $dS_u = \sigma d W_u$
- - initial value, S_t = s, W_t is standard 1D Brownian motion,
- - \sigma is constant
3. agent can not affect drift or autocorrelation structure of stock
4. 

global:
 - gamma = discount price?
 - sigma = brownian motion variance
 - terminal time T at which stock dies or sumn

'frozen inventory' strategy
inactive trader
no limit orders and only holds inventory of q stocks until T
value function given by v(x,s,q,t), where
 - x = initial wealth, q = nstocks, t = time
 - s = initial stock value, = midprice
"""


gamma = sigma = 1; T = 1000
# v(x,s,q,t), 
# x = initial wealth, s = initial stock value, q = nstocks, t = time
def frozen_value(initial_wealth, stock_val, nstocks, time):
    first = -np.exp(-gamma*(initial_wealth+nstocks*stock_val))
    second = np.exp((gamma*nstocks*sigma)**2 * (T - time) / 2)
    return first * second

"""
reservation_bid is price that makes agent indifferent to buy a stock
v(x-r^b(s,q,t), s, q+1, t) >= v(x,s,q,t)
reservation_ask is price that makes agent indifferent to sell a stock
v(x+r^a(s,q,t), s, q-1, t) >= v(x,s,q,t)
where r^b, r^a is bid, ask price
"""
def res_ask_price(s,q,t):
    return s + (1-2*q) * gamma * sigma**2 * (T-t)

def res_bid_price(s,q,t):
    return s - (1+2*q) * gamma * sigma**2 * (T-t)

# avg between bid and ask
def res_price(s, q, t):  # reservation / indifference price
    return s - q * gamma * sigma**2 * (T-t)


"""
2.4 :  adding limit orders

quotes bid price p^b, ask price p^a
focus on distances \delta^b = s - p^b and \delta^a = p^a - s

imagine market order of Q stocks arrives, the Q limit orders with lowest 
ask prices are sold. if p^Q is price of highest limit order, 
\Delta p = p^Q - s  is the temporary market impact of the trade

ASSUME 
- market buy orders "lift" agent's limit asks at Poisson rate
\lambda^a(\delta^a), monotonically decreasing function
- markey sells will "hit" the buy limit orders at rate \lambda^b
these rates are also called the Poisson "intensities"

X = wealth of agent. N_t^a is # stocks sold, $N_t^b$ is # stocks bought
$dX_t = p^a dN_t^a - p^b dN_t^b$

# stocks held at time $t$ is q_t
"""

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
    def __init__(self, inventory, wealth):
        self.I = inventory
        self.W = wealth
        self.book = OrderBook()
        self.actions = ['buy', 'sell', 'bid', 'ask']
        # --- market order parameters --- #
        # assume environment only does market orders
        # rate of market orders
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 20
        self.Lambda_s = 20
        self.K_b = 1
        self.K_s = 1
        # action stuff
        self.sigma = 1e-2
        self.gamma = 1
        self.terminal_time = 1  # second
        self.dt = 5e-3   # millisecond

        # --- neural network --- #
        # predict actions from observations

        # variables = (highest_bid, lowest_ask, n_highest_bid, n_lowest_ask, wealth[t-1], inventory[t-1], midprices[t-1])
        obs_dim = 7
        hidden_dim = 10
        # learn the expectation of the Q function
        self.value_network = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                           nn.LeakyReLU(),
                                           nn.Linear(hidden_dim, 1),
                                           Exponential())
        self.optimizer = torch.optim.Adam(self.value_network.parameters(), lr=1e-3)
        act_dim = 6
        self.policy_network = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(hidden_dim, act_dim))
    
    def act(self, action: int, nstocks: int, price=0.):
        if action == 0:  #'buy'
            n_bought, bought = self.book.buy(nstocks)
            return bought
        elif action == 1:  #'sell':
            n_sold, sold = self.book.sell(nstocks)
            return sold
        if not price:
            raise ValueError("Price must be specified for 'bid' (1) and 'ask' (2) actions")
        if action == 2: # 'bid':
            self.book.bid(nstocks, price)
        elif action == 3:  #'ask':
            self.book.ask(nstocks, price)
        else:
            raise ValueError(f"Invalid action {action}. Action is 0 for market-buy, 1 for market-sell, 2 for limit-bid, and 3 for limit-ask.")
    
    # trading intensities
    def lambda_buy(self, delta_a):
        k = self.alpha * self.K_b
        A = self.Lambda_b / self.alpha
        return A * np.exp(-k*delta_a)

    def lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * np.exp(-k*delta_b)
    
    # objective with no inventory dynamics?
    def frozen_value(self, initial_wealth, stock_val, nstocks, time):
        first = -np.exp(-gamma*(initial_wealth + nstocks*stock_val))
        second = np.exp((gamma*nstocks*sigma)**2 * (self.terminal_time - time) / 2)
        return first * second
    
    def step(self):
        # initial state of order book
        initial_midprice = 100
        initial_spread = 10
        initial_num_stocks = 100
        self.book.bid(initial_num_stocks//2, initial_midprice-initial_spread/2)
        self.book.ask(initial_num_stocks-initial_num_stocks//2, initial_midprice+initial_spread/2)
        # random event of market order
        n_times = int(self.terminal_time / self.dt)
        wealth = np.zeros(n_times+1)
        wealth[0] = self.wealth
        inventory = np.zeros(n_times+1)
        inventory[0] = self.n
        midprices = np.zeros(n_times+1)
        midprices[0] = initial_midprice
        epsilon = 0

        for t in range(1,n_times+1):
            # take action after observing state

            # --- MARKET MAKER ACTIONS --- #
            # limit orders
            # --- INPUTS --- #
            mid = midprices[t-1]

            delta_b = self.book.delta_b
            delta_a = self.book.delta_a
            if not len(self.book.bids) or not len(self.book.asks):
                break
            n_highest_bid = self.book.bids[0][1]
            n_lowest_ask = self.book.asks[0][1]
            lowest_ask  = self.book.midprice + delta_a
            highest_bid = self.book.midprice - delta_b
            #obs_state = (highest_bid, lowest_ask, n_highest_bid, n_lowest_ask)
            # want to change obs_state by taking these actions

            # potentially add n_buy, n_ask to state
            act_state = (n_bid, bid_price, n_ask, ask_price)

            # OBSERVE STATE
            observations = (highest_bid, lowest_ask, n_highest_bid, n_lowest_ask, wealth[t-1], inventory[t-1], midprices[t-1])
            
            # PERFORM ACTION

            action = self.policy_network()

            # OBSERVE NEXT STATE
            actions = (highest_bid, lowest_ask, n_highest_bid, n_lowest_ask)

            variables = (highest_bid, lowest_ask, n_highest_bid, n_lowest_ask, wealth[t-1], inventory[t-1], midprices[t-1])
            vars = torch.tensor(variables)
            obs_dim = len(variables)
            act_dim = len(act_state)
            hidden_dim = 10
            # learn the expectation of the Q function
            self.value_network = nn.Sequential(nn.Linear(obs_dim, hidden_dim), 
                                               nn.LeakyReLU(),
                                               nn.Linear(hidden_dim, 1))
            # input: midprice, delta_b, delta_a
            # more input: self.book.bids[0][1], self.book.asks[0][1]
            # output: bid_price, bid_num, ask_price, ask_num

            # --- OUTPUTS --- #
            bid_price = np.random.normal(mid - delta_b, delta_b/4)
            ask_price = np.random.normal(mid + delta_a, delta_a/4)
            # makes sure that bid price is less than ask price always?
            # only needed for larger variance in the prices that may cross
            bid_price = np.clip(bid_price, 0, min(mid + delta_a, ask_price))
            ask_price = np.clip(ask_price, max(bid_price, mid - delta_b), np.inf)
            #self.book.plot()
            n_bid = np.random.poisson(n_highest_bid)
            n_ask = np.random.poisson(n_lowest_ask)
            self.book.bid(n_bid, bid_price)
            self.book.ask(n_ask, ask_price)


            # ---- transition to next state ---- #

            # --- market act --- #
            # get random number of buys, sells
            nbuy  = np.random.poisson(self.lambda_buy(delta_a))
            nsell = np.random.poisson(self.lambda_sell(delta_b))
            n_ask_lift, bought = self.book.buy(nbuy)
            n_bid_hit, sold = self.book.sell(nsell)

            # update wealth
            # make money from people buying, lose money from people selling
            wealth[t] = wealth[t-1] + bought - sold
            inventory[t] = inventory[t-1] + n_bid_hit - n_ask_lift

            # scaled random walk
            new_eps = np.random.normal(0,self.dt*t)
            time_drift = self.dt*t*5
            new_midprice = midprices[t-1]*np.exp(time_drift + self.sigma*(new_eps - epsilon))
            # clip midprice so that it is within bid, ask spread
            midprices[t] = np.clip(new_midprice, self.book.midprice - delta_b, self.book.midprice + delta_a)
            epsilon = new_eps

            # update expecation based on t-0 network
            new_val = -torch.from_numpy(np.exp(-self.gamma(wealth[t]+inventory[t]*midprices[t])))
            pred_val = self.value_network(vars)
            loss = -(new_val - pred_val)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # plot everything!
        self.book.plot()
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        axs[2].set(xlabel="Time")
        axs[0].set(ylabel='Wealth')
        axs[1].set(ylabel='Inventory')
        axs[2].set(ylabel='Midprice')
        axs[0].plot(wealth)
        axs[0].plot(wealth + inventory*midprices,label='Total Value')
        axs[1].plot(inventory)
        axs[2].plot(midprices)
        axs[0].legend()
        plt.show()


# tuple of time, wealth_diff, inventory_diff, midprice_diff
# from https://discovery.ucl.ac.uk/id/eprint/10116730/1/RLforHFMM.pdf 
def step_reward(output, a=1, b=1):
    wealth_diff, inventory_diff, time_diff = output
    return a*wealth_diff + np.exp(-b*time_diff) * np.sign(inventory_diff)

def train_market():
    """ Monte-Carlo Ish Thing """
    # initialize parameters

    dt = 0.005
    # initialize POLICY function
    # some neural network that maps states to actions
    # initialize policy to AV-STOICKov policy? train NN to learn this function lol

    # initialize VALUE function
    # Q-value function

    num_episodes = 1000
    for episode in range(num_episodes):
        # run trajectories
        mm = MarketMaker(100, 1000, dt=dt)

        # move market forward 30 times
        # randomly initialize order book
        # do random market orders and random limit orders

        # run full trajectory, given current policy / state
        num_timesteps = 10000  # define horizon
        times = np.arange(0, num_timesteps*dt, dt)
        trajectory_return = 0
        trajectory = []
        for t in times:
            # observe state
            # TUPLE of (delta_a, delta_b, n_high_bid, n_low_ask)
            state = mm.observe_state()
            # observe and take action
            # TUPLE of (n_bid, bid_price, n_ask, ask_price)
            action = mm.act(state)
            # step through one market dynamics evolution
            # -- perform random market orders
            # -- jump to next timestep
            # and record change in wealth, inventory, midprice? 
            output = mm.step()
            # step return
            trajectory_return += step_reward(output)

            trajectory.append(state, action)  # store t as well?
        
        # observe final state and reward
        trajectory_return += mm.final_reward()

        # --- UPDATE POLICY AND VALUE FUNCTIONS --- #

        # backpropagate through trajectory to define reward from t?

        # use advantage function to debias 
        advantage = trajectory_return - mm.Q_function(trajectory)
        # compute discounted advantage using discounted returns?
        # backstep
        

        # RUN PPO


        
        

        

        











if __name__ == "__main__":
    mm = MarketMaker(0, 0)
    mm.step()
