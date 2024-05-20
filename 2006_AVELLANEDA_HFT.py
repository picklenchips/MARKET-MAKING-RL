from limit_order_book import OrderBook, LimitOrder
from util import uFormat, mpl, plt, np

"""
@article{2006_avellaneda,
    author = {Marco Avellaneda and Sasha Stoikov},
    title = {High-frequency trading in a limit order book},
    doi = {10.1080/14697680701381228},
    journal = {Quantitative Finance},
    number = {3},
    pages = {217--224},
    publisher = {Routledge},
    url = {https://doi.org/10.1080/14697680701381228},
    volume = {8},
    year = {2008},
    bdsk-url-1 = {https://doi.org/10.1080/14697680701381228}}
"""

# asssume money market pays no interest
# mid-price given by $dS_u = \sigma d W_u$
"""
ASSUMPTIONS:
1. asssume money market pays no interest
2. mid-price given by $dS_u = \sigma d W_u$
- - initial value, S_t = s, W_t is standard 1D Brownian motion,
- - \sigma is constant
3. agent can not affect drift or autocorrelation structure of stock
4. 
"""

"""
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

class MarketMaker():
    def __init__(self, nstocks, wealth):
        self.n = nstocks
        self.wealth = wealth
        self.book = OrderBook()
        self.actions = ['buy', 'sell', 'bid', 'ask']
        # --- market order parameters --- #
        # alpha is 1.53 for US stocks and 1.4 for NASDAQ stocks
        self.alpha = 1.53
        self.Lambda_b = 20
        self.Lambda_s = 20
        self.K_b = 1
        self.K_s = 1
        # action stuff
        self.sigma = 1
        self.gamma = 1
        self.terminal_time = 1000  # second
        self.dt = 1   # millisecond

    
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
        for t in range(1,n_times+1):
            mid = self.book.midprice
            delta_b = self.book.delta_b
            delta_a = self.book.delta_a
            # take action after observing state
            bid_price = np.random.normal(mid - delta_b, delta_b/4)
            ask_price = np.random.normal(mid + delta_a, delta_a/4)
            # makes sure that bid price is less than ask price always?
            # only needed for larger variance in the prices that may cross
            #bid_price = np.clip(bid_price, 0, min(mid + delta_a, ask_price))
            #ask_price = np.clip(ask_price, max(bid_price, mid - delta_b), np.inf)
            #self.book.plot()
            n_highest_bid = self.book.bids[0][1]
            n_lowest_ask = self.book.asks[0][1]
            n_bid = np.random.poisson(n_highest_bid)
            n_ask = np.random.poisson(n_lowest_ask)
            self.book.bid(n_bid, bid_price)
            self.book.ask(n_ask, ask_price)
            # --- MARKET ACTIONS --- #
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
            midprices[t] = self.book.midprice

        self.book.plot()
        fig, axs = plt.subplots(3,1, figsize=(10,8))
        axs[0].set_title("Wealth")
        axs[1].set_title("Inventory")
        axs[2].set_title("Midprice") 
        axs[2].set(xlabel="Time")
        axs[1].set(ylabel='Number of Stocks')
        axs[0].plot(wealth)
        axs[0].plot(wealth + inventory*midprices)
        axs[1].plot(inventory)
        axs[2].plot(midprices)
        plt.show()


if __name__ == "__main__":
    mm = MarketMaker(0, 0)
    mm.step()
