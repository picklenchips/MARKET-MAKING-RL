import numpy as np
import heapq  # priority queue
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
#           |   Red  |   Blue  |  Orange |  Purple | Yellow  |   Green |   Teal  | Grey
hexcolors = ['DC267F', '648FFF', 'FE6100', '785EF0', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])
from util import uFormat

# 
class LimitOrder():
    def __init__(self, nstocks, price):
        self.n = nstocks
        self.p = price

# assume midprice is just average between two sell orders? 
# that is, not weighted by number of things in each order
# assume that limit orders can only be bought and sold by market orders
# that is, we ignore the case that a buy and sell limit order have the same 
# exact price

# should we change buying and selling to be the same action, but 
# with a negative price??

#TODO: any use case for returning the bought and sold arrays for market orders?
#      : could just change to total price spent or something
#TODO: change buying and selling to be the same action?
class OrderBook():
    def __init__(self):
        # keep track of limit orders
        self.buys = []
        self.sells = []
        self.lowest_sell = np.inf
        self.highest_buy = 0
        # and their relevant pricing dynamics
        self.midprice = None
        self.spread = None
    
    def recalculate(self):
        """ Recalculate self.midprice and self.spread """
        self.spread = self.midprice = None 
        if len(self.sells):
            self.midprice = self.sells[0][0]
            if len(self.buys):
                lowest_sell = self.sells[0][0]
                highest_buy = -self.buys[0][0]
                self.midprice = (lowest_sell+highest_buy)/2
                self.spread = lowest_sell-highest_buy
                if self.spread < 0:
                    print("ERROR: unrealistic spread!!")
        elif len(self.buys):
            self.midprice = -self.buys[0][0]

    def market_buy(self, nstocks, maxprice=None):
        """Buy (up to) nstocks stocks to lowest-priced limit-sell orders
        returns tuple of int: num stocks bought, 
                         list of (price, n_stocks) orders that have been bought
        optionally, only buy stocks valued below max price"""
        bought = []; n_bought = 0; do_update = False
        while nstocks > 0 and len(self.sells):
            if maxprice:  # only buy stocks less than max price
                if self.sells[0][0] > maxprice: break
            # buy up to nstocks stocks from the n available at this price
            price, n = heapq.heappop(self.sells)
            n_bought += min(n, nstocks)
            bought.append((price, min(n, nstocks)))
            # place remaining portion of limit order back
            if nstocks < n: 
                heapq.heappush(self.sells, (price, n-nstocks))
            else: 
                do_update = True
            nstocks -= n
        if do_update: self.recalculate()
        return n_bought, bought

    def market_sell(self, nstocks, minprice=None):
        """Sell (up to) nstocks stocks to highest-priced limit-buy orders
        optionally, only sell stocks valued above min price"""
        sold = []; n_sold = 0; do_update = False
        while nstocks > 0 and len(self.buys):
            if minprice:  # only sell stocks greater than min price
                if -self.buys[0][0] < minprice: break
            price, n = heapq.heappop(self.buys)
            price = -price  # convert back to normal price
            # sell up to nstocks stocks to the n available at this price
            n_sold += min(n, nstocks)
            sold.append((price,min(n, nstocks)))
            # place remaining portion of limit order back
            if nstocks < n: 
                heapq.heappush(self.buys,(-price, n-nstocks))
            else: 
                do_update = True
            nstocks -= n
        if do_update: self.recalculate()
        return n_sold, sold

    def limit_buy(self, nstocks, price):
        """Add a limit-buy order. Sorted highest-to-lowest"""
        # buying higher than lowest sell -> market buy instead
        if len(self.sells):
            if price >= self.sells[0][0]:
                nbought, bought = self.market_buy(nstocks, maxprice=price)
                nstocks -= nbought
                if nstocks == 0: return  # all eaten!
        heapq.heappush(self.buys, (-price, nstocks))
        # is now highest buy order, recalculate
        if -price == self.buys[0][0]:
            self.recalculate()

    def limit_sell(self, nstocks, price):
        """Add a limit-sell order"""
        # selling lower than highest buy order -> sell some now!
        if len(self.buys):
            if price <= -self.buys[0][0]:
                nsold, sold = self.market_sell(nstocks, minprice=price)
                nstocks -= nsold
                if nstocks == 0: return  # all eaten!
        heapq.heappush(self.sells, (price, nstocks))
        # is now lowest sell order, recalculate
        if price == self.sells[0][0]:
            self.recalculate()
    
    def plot(self):
        """Make histogram of current limit-orders"""
        fig, ax = plt.subplots()
        ax.set(xlabel='Price per Share',ylabel='Volume')
        ax.bar([-b[0] for b in self.buys],[b[1] for b in self.buys],label='Buy Orders',edgecolor='black',linewidth=0.5)
        ax.bar([s[0] for s in self.sells],[s[1] for s in self.sells],label='Sell Orders',edgecolor='black',linewidth=0.5)
        if self.midprice:
            lowx, highx = ax.get_ylim()
            ax.plot([self.midprice,self.midprice],[lowx,highx*0.8],linestyle='dashed',color='black')
        plt.title(f"Order Book. midprice = {uFormat(self.midprice,0)}, spread = {uFormat(self.spread,0)}")
        plt.legend()
        plt.show()

# we getting Pythonic up in this
if __name__ == "__main__":
    # test orderbook eating properties
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
        book.limit_buy(n,buy_orders[i])
        n = np.random.randint(1,10)
        print(f'placing sell order of (${sell_orders[i]},{n}). ')
        book.limit_sell(n,sell_orders[i])
        book.plot()