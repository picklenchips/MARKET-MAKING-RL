from util import uFormat, mpl, plt, np
import heapq  # priority queue

"""
Follow the naive approach taken in 
https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf 
to define the order book environment
"""

class LimitOrder():
    def __init__(self, nstocks, price):
        self.n = nstocks
        self.p = price

# should we change buying and selling to be the same action, but 
# with a negative price??

class OrderBook():
    """
    Creates a limit-order book with four distinct actions:
    - buy: market-buy # of stocks cheaper than some maximum price
    - sell: market-sell # of stocks more expensive than some minimum price 
    - bid: create a limit order to buy # stocks at some price
      - stored self.bids as (-price, #)
    - ask: create a limit order to sell # stocks at some price 
      - stored self.asks as (price, #)
    and two states self.midprice, self.spread
    """
    def __init__(self):
        # keep track of limit orders
        self.bids = []
        self.asks = []
        # and their relevant pricing dynamics
        self.midprice = None
        self.spread = None
        self.delta_b = None
        self.delta_a = None
    
    def recalculate(self):
        """ Recalculate self.midprice and self.spread """
        self.spread = self.midprice = None 
        if len(self.asks):
            self.midprice = self.asks[0][0]
            if len(self.bids):
                lowest_sell = self.asks[0][0]
                highest_buy = -self.bids[0][0]
                self.midprice = (lowest_sell+highest_buy)/2
                self.spread = lowest_sell-highest_buy
                if self.spread < 0:
                    print("ERROR: unrealistic spread!!")
        elif len(self.bids):
            self.midprice = -self.bids[0][0]

    def buy(self, nstocks, maxprice=None):
        """Buy (up to) nstocks stocks to lowest-priced limit-sell orders
        returns tuple of int: num stocks bought, 
                         list of (price, n_stocks) orders that have been bought
        optionally, only buy stocks valued below max price"""
        bought = []; n_bought = 0; do_update = False
        while nstocks > 0 and len(self.asks):
            if maxprice:  # only buy stocks less than max price
                if self.asks[0][0] > maxprice: break
            # buy up to nstocks stocks from the n available at this price
            price, n = heapq.heappop(self.asks)
            n_bought += min(n, nstocks)
            bought.append((price, min(n, nstocks)))
            # place remaining portion of limit order back
            if nstocks < n: 
                heapq.heappush(self.asks, (price, n-nstocks))
            else: 
                do_update = True
            nstocks -= n
        if do_update: self.recalculate()
        return n_bought, bought

    def sell(self, nstocks, minprice=None):
        """Sell (up to) nstocks stocks to highest-priced limit-buy orders
        optionally, only sell stocks valued above min price"""
        sold = []; n_sold = 0; do_update = False
        while nstocks > 0 and len(self.bids):
            if minprice:  # only sell stocks greater than min price
                if -self.bids[0][0] < minprice: break
            price, n = heapq.heappop(self.bids)
            price = -price  # convert back to normal price
            # sell up to nstocks stocks to the n available at this price
            n_sold += min(n, nstocks)
            sold.append((price,min(n, nstocks)))
            # place remaining portion of limit order back
            if nstocks < n: 
                heapq.heappush(self.bids,(-price, n-nstocks))
            else: 
                do_update = True
            nstocks -= n
        if do_update: self.recalculate()
        return n_sold, sold

    def bid(self, nstocks, price):
        """Add a limit-buy order. Sorted highest-to-lowest"""
        # buying higher than lowest sell -> market buy instead
        if len(self.asks):
            if price >= self.asks[0][0]:
                nbought, bought = self.buy(nstocks, maxprice=price)
                nstocks -= nbought
                if nstocks == 0: return  # all eaten!
        heapq.heappush(self.bids, (-price, nstocks))
        # is now highest buy order, recalculate
        if -price == self.bids[0][0]:
            self.recalculate()

    def ask(self, nstocks, price):
        """Add a limit-sell order"""
        # selling lower than highest buy order -> sell some now!
        if len(self.bids):
            if price <= -self.bids[0][0]:
                nsold, sold = self.sell(nstocks, minprice=price)
                nstocks -= nsold
                if nstocks == 0: return  # all eaten!
        heapq.heappush(self.asks, (price, nstocks))
        # is now lowest sell order, recalculate
        if price == self.asks[0][0]:
            self.recalculate()
    
    def plot(self):
        """Make histogram of current limit-orders"""
        fig, ax = plt.subplots()
        ax.set(xlabel='Price per Share',ylabel='Volume')
        ax.bar([-b[0] for b in self.bids],[b[1] for b in self.bids],label='Bids',edgecolor='black',linewidth=0.5)
        ax.bar([s[0] for s in self.asks],[s[1] for s in self.asks],label='Asks',edgecolor='black',linewidth=0.5)
        title = "Order Book"
        if self.midprice:  # add dashed line for midprice
            lowx, highx = ax.get_ylim()
            ax.plot([self.midprice,self.midprice],[lowx,highx*0.8],linestyle='dashed',color='black')
            title += f" - midprice = {uFormat(self.midprice,0)}, spread = {uFormat(self.spread,0)}"
        plt.title(title)
        plt.legend(loc='upper center',ncol=2)
        plt.show()


# we getting Pythonic up in this
if __name__ == "__main__":
    # test orderbook eating properties
    while 1:
        N = input("number of buy-sell iterations:\n>> ").strip()
        if N.isdigit():
            N = int(N)
        else: 
            N = 10
        mean_buy = 100
        mean_sell = 100
        std_buy = 30
        std_sell = 30
        buy_orders = (np.random.normal(size=(N,))*std_buy + mean_buy).astype(int)
        sell_orders = (np.random.normal(size=(N,))*std_sell + mean_sell).astype(int)

        book = OrderBook()

        # alternate buy, sell orders
        for i in range(N):
            n = np.random.randint(1,10)
            print(f'placing buy order of (${buy_orders[i]},{n}). ')
            book.bid(n,buy_orders[i])
            n = np.random.randint(1,10)
            print(f'placing sell order of (${sell_orders[i]},{n}). ')
            book.ask(n,sell_orders[i])
        book.plot()