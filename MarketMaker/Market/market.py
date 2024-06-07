
from MarketMaker.util import np, np2torch, torch
from MarketMaker.Market.limit_order_book import OrderBook as LOB
from MarketMaker.config import Config

class BaseMarket():
    """ Market environment """
    def __init__(self, inventory: int, wealth: float, config: Config):
        self.config = config
        self.I = inventory
        self.W = wealth
        # update book later using self.init_book()
        self.book = LOB()
    # --- market order parameters --- #
        # OLD
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 20  # average "volume" of market orders at each step
        self.Lambda_s = 20  
        self.K_b = 1  # as K increases, rate w delta decreases
        self.K_s = 1 
        self.betas = (7.0187, -0.3265, 0.0554, 3.27, 1.16, 0.078)  # ryan's new model
        # action stuff
        self.sigma = config.sigma
        self.gamma = config.gamma 
        self.max_t = config.max_t  # second
        self.dt    = config.dt   # millisecond
        # reward stuff
        # max possible wealth change from one market step
        max_dW = self.book.midprice * 1200
        self.a = 1/max_dW  # how much we weigh dW
        self.b = 0.1/self.max_t   # time-weighted inventory change
        self.c = 1/self.max_t     # how much we negatively weigh time
        self.discount = config.discount

    def reset(self, mid=100, spread=10, nstocks=100000, nsteps=1000, substeps=1, 
              make_bell=True, plot=False):
        """ Randomly initialize order book 
        - nstocks is num stocks on each side """
        if self.book: del(self.book)
        self.book = LOB(mid)
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
            if not plot:
                return
        # start with symmetric spread and market-step
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
                title = f"{t}:{action}: {tuple(round(a,2) for a in old_state)}->{tuple(round(a,2) for a in state)}"
                old_book.plot(2, title)
                break
            old_book = self.book.copy()
            old_state = self.state()
            action = self.act(state)
            if state[0] == 0 or state[2] == 0:
                old_book.recalculate()
                action = tuple(round(a,2) for a in action)
                title = f"{t}:{action}: {tuple(round(a,2) for a in old_state)}->{tuple(round(a,2) for a in state)}"
                old_book.plot(2, title)
                break
        self.book.plot()
            
    def is_empty(self):
        return self.book.is_empty()
    
# --- ENVIRONMENT / DYNAMICS --- #
    def lambda_buy(self, delta, q):
        if not q: return 0
        lambdaa =  np.exp(self.betas[0]+self.betas[1]*np.log(delta)+self.betas[2]*np.log(delta)**2+self.betas[3]*np.log(1+q)+self.betas[4]*np.log(1+q)**2+self.betas[5]*np.log(delta+1+q))
        print(delta, q, lambdaa)
        return lambdaa
    
    def lambda_sell(self, delta, q):
        return self.lambda_buy(delta, q)
    
    def step(self, nsteps=1, plot=False):
        """ Evolve market order book by updating midprice and placing market orders
        - returns change in wealth, inventory (dW, dI) """
        dW = dI = 0
        for step in range(nsteps):
            self.book.update_midprice()  # STEP MIDPRICE
            try:
                lambda_buy = self.lambda_buy(self.book.delta_a, self.book.nlow_ask)
                lambda_sell = self.lambda_sell(self.book.delta_b, self.book.nhigh_bid)
                nbuy  = np.random.poisson(lambda_buy)
                nsell = np.random.poisson(lambda_sell)
            except ValueError:
                self.book.print_state()
                if plot:
                    return dW, dI, self.book.midprice, (nsell, self.book.high_bid, nbuy, self.book.low_ask)
                return 0, 0, self.book.midprice
            n_ask_lift, bought = self.book.buy(nbuy)
            n_bid_hit, sold = self.book.sell(nsell)  # 
            dW += bought - sold; dI += n_bid_hit - n_ask_lift
        if plot:  # for limit order plotting purposes
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
        - returns action taken, (n_bid, delta_b, n_ask, delta_a)
        - naive sets price variance of 'stupid' policy taken around midprice
        """
        try:  # invalid input action??
            if len(state) < 3: raise TypeError
        except TypeError:  # resample state
            state = self.state()
        if len(state) == 4:  # default "naive" policy
            rate = 3/4
            n_bid, delta_b, n_ask, delta_a = state
            delta_b_new = np.random.normal(delta_b, delta_b/4)
            delta_a_new = np.random.normal(delta_a, delta_a/4)
            n_bid = np.random.poisson(n_bid*rate)
            n_ask = np.random.poisson(n_ask*rate) 
            action = (n_bid, delta_b_new, n_ask, delta_a_new)
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
        delta_b = max(delta_b, 0)  # deltas cant be negative
        delta_a = max(delta_a, 0)
        # can only bid integer multiples
        n_bid = round(n_bid)
        n_ask = round(n_ask)
        self.book.bid(n_bid, self.book.midprice - delta_b)
        self.book.ask(n_ask, self.book.midprice + delta_a)

if __name__ == "__main__":
    """ 
    Test
    Tha 
    Market
    """
    config = Config()
    M = BaseMarket(0, 0, config)
    for i in range(20):
        M.reset(plot=True, make_bell=True)