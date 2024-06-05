from limit_order_book import OrderBook
from util import np, np2torch
import torch
from glob import glob
from config import Config

class Market():
    """ Market environment """
    def __init__(self, inventory: int, wealth: float, config: Config):
        self.config = config
        self.I = inventory
        self.W = wealth
        # update book later using self.init_book()
        self.book = OrderBook()
    # --- market order parameters --- #
        # OLD
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 20  # average "volume" of market orders at each step
        self.Lambda_s = 20  
        self.K_b = 1  # as K increases, rate w delta decreases
        self.K_s = 1 
        self.betas = (7.40417, -3.12017, 0.167814)  # ryan's new model
        # action stuff
        self.sigma = config.sigma
        self.gamma = config.gamma 
        self.max_t = config.max_t  # second
        self.dt    = config.dt   # millisecond
        # reward stuff
        self.a = 1  # how much we weigh dW
        self.b = 0.1  # how much we weigh dI
        self.c = 1   # how much we negatively weigh time
        self.discount = config.discount

    def reset(self, mid=100, spread=10, nstocks=10000, nsteps=1000, substeps=1, 
              make_bell=True, plot=False):
        """ Randomly initialize order book 
        - nstocks is num stocks on each side """
        if self.book: del(self.book)
        self.book = OrderBook(mid)
        # start with symmetric spread
        # JUST DO A BELL CURVE ON EACH SIDE
        tot_amount = 0
        if make_bell:
            while tot_amount < nstocks:
                delta_b = np.random.normal(spread/2, spread/4)
                delta_a = np.random.normal(spread/2, spread/4)
                nbid = np.random.poisson(2*nstocks/nsteps)
                nask = np.random.poisson(2*nstocks/nsteps)
                self.book.bid(nbid, mid-delta_b)
                self.book.ask(nask, mid+delta_a)
                tot_amount += nask + nbid
            if plot:
                self.book.plot(title=', '.join(map(str, self.state())))
            return
        # start with symmetric spread and market-step
        self.book.bid(nstocks//2, mid-spread/2)
        self.book.ask(nstocks//2, mid+spread/2)
        for t in range(nsteps):
            # perform random MARKET ORDERS
            old_book = self.book.copy()
            old_state = self.state()
            dW, dI, mid = self.step(substeps)
            state  = self.state()
            if state[0] == 0 or state[2] == 0:
                print("OOF: ",state)
            # perform random LIMIT ORDERS using naive action
            state = self.state()
            if state[0] == 0 or state[2] == 0:
                old_book.recalculate()
                action = tuple(round(a,2) for a in action)
                title = f"{t}:{action}: {old_state}->{state}"
                old_book.plot(title)
                break
            old_book = self.book.copy()
            old_state = self.state()
            action = self.act(state)
            if state[0] == 0 or state[2] == 0:
                old_book.recalculate()
                action = tuple(round(a,2) for a in action)
                title = f"{t}:{action}: {old_state}->{state}"
                old_book.plot(title)
                break
        self.book.plot()
            
    def is_empty(self):
        return self.book.is_empty()
    
# --- ENVIRONMENT / DYNAMICS --- #
#TODO: implement the latest version of ryans thing 
# that takes in the quantities instead of delta_a, delta_b
    def avellaneda_lambda_buy(self, delta_a):
        k = self.alpha * self.K_b
        A = self.Lambda_b / self.alpha
        return A * np.exp(-k*delta_a)
    
    def lambda_buy(self, q):
        if not q: return 0
        return np.exp(self.betas[0]+self.betas[1]*np.log(1+q)+self.betas[2]*np.log(1+q)**2)
    
    def lambda_sell(self, q):
        return self.lambda_buy(q)

    def avellaneda_lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * np.exp(-k*delta_b)
    
    def step(self, nsteps=1, plot=False):
        """ Evolve market order book by updating midprice and placing market orders
        - returns change in wealth, inventory (dW, dI) """
        dW = dI = 0
        for step in range(nsteps):
            self.book.update_midprice()  # STEP MIDPRICE
            nbuy  = np.random.poisson(self.lambda_buy(self.book.nlow_ask))
            nsell = np.random.poisson(self.lambda_sell(self.book.nhigh_bid))
            n_ask_lift, bought = self.book.buy(nbuy)
            n_bid_hit, sold = self.book.sell(nsell)  # 
            dW += bought - sold; dI += n_bid_hit - n_ask_lift
        if plot:  # for limit order plotting purposes
            if not n_bid_hit: n_bid_hit = 1
            if not n_ask_lift: n_ask_lift = 1
            return dW, dI, self.book.midprice, (n_bid_hit, self.book.high_bid, n_ask_lift, self.book.low_ask)
        return dW, dI, self.book.midprice

# --- STATES / ACTIONS --- #
    def state(self) -> tuple[int, float, int, float]:
        """ returns tuple of n_bid, bid_price, n_ask, ask_price """
        return self.book.nhigh_bid, self.book.delta_b, self.book.nlow_ask, self.book.delta_a

    def act(self, state: tuple | np.ndarray | list | torch.Tensor, policy=None):
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
            state = self.state()
        if len(state) == 4:  # default "naive" policy
            rate = 3/4
            delta_b = self.book.delta_b; delta_a = self.book.delta_a
            n_bid, bid_price, n_ask, ask_price = state
            bid_price = np.random.normal(bid_price, delta_b/4)
            ask_price = np.random.normal(ask_price, delta_a/4)
            n_bid = np.random.poisson(n_bid*rate)
            n_ask = np.random.poisson(n_ask*rate) 
            action = (n_bid, bid_price, n_ask, ask_price)
        elif len(state) == 3:  # avellaneda policy
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
            action = (n_bid, self.book.midprice - bid_price, n_ask, ask_price - self.book.midprice)
        else:  # just run the network on the state!
            if not policy:
                raise NotImplementedError("Policy network is not defined!")
            action = policy(state)
        self.submit(*action)
        return action
    
    def submit(self, n_bid, delta_b, n_ask, delta_a):
        """ Perform a limit order action, making (up to) 2 limit orders: a bid and ask
        - if n_bid or n_ask < 0.5, will not bid or ask """
        # make sure that bid price is always less than ask price
        delta_b = max(delta_b, 0)  # deltas cant be negative
        delta_a = max(delta_a, 0)
        bid_price = self.book.midprice - delta_b
        ask_price = self.book.midprice + delta_a
        #bid_price = np.clip(bid_price, 0, self.book.low_ask)
        #ask_price = np.clip(ask_price, self.book.high_bid, np.inf)
        # can only bid integer multiples
        n_bid = round(n_bid)
        n_ask = round(n_ask)
        # LOB automatically only inserts price in cents so dw ab
        # bid_price = round(bid_price.item(), 2)
        self.book.bid(n_bid, bid_price)
        self.book.ask(n_ask, ask_price)
        #return n_bid, bid_price, n_ask, ask_price

# --- REWARD FUNCTIONS --- #
    def frozen_reward(self, initial_wealth, stock_val, nstocks, time_left):
        first = -np.exp(-self.gamma*(initial_wealth + nstocks*stock_val))
        second = np.exp((self.gamma*nstocks*self.sigma)**2 * time_left / 2)
        return first * second

    def reward(self, r_state):
        """ immediate reward """
        # dW, dI, time_left = reward_state
        if not self.config.immediate_reward: 
            return 0
        if isinstance(r_state, tuple):
            r_state = np.array(r_state)
        if self.config.subtract_time:
            return self.a*r_state[...,0] + np.exp(-self.b*r_state[...,2]) * np.sign(r_state[...,1]) - self.c*r_state[...,2]
        return self.a*r_state[...,0]# + np.exp(-self.b*r_state[...,2]) * np.sign(r_state[...,1])
    
    def final_reward(self, W, inventory, midprice):
        return W + inventory*midprice

    def reservation_price(self, midprice, inventory, t_left):  # reservation / indifference price
        return midprice - inventory * self.gamma * self.sigma**2 * t_left

    def optimal_spread(self, time_left):
        return self.gamma * self.sigma**2 * time_left + 2*np.log(1+2*self.gamma/(self.alpha*(self.K_b+self.K_s)))/self.gamma


# test market initialization 
# yuh
# 
# uh 
# oh ?
if __name__ == "__main__":
    config = Config()
    M = Market(0, 0, config)
    for i in range(20):
        M.reset(plot=True, make_bell=True)