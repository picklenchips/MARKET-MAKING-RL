from util import uFormat, mpl, plt, np
from util import FIGSIZE, SAVEDIR, SAVEEXT
import heapq  # priority queue
from stochastic.processes.continuous.brownian_motion import BrownianMotion
import math  # for ceil

class OrderBook():
    """
    Creates a limit-order book with four distinct actions:
    - buy: market-buy # of stocks cheaper than some maximum price
    - sell: market-sell # of stocks more expensive than some minimum price 
    - bid: create a limit order to buy # stocks at some price
      - stored self.bids as (-price, #)
      - highest bid tracked as (self.high_bid, self.nhigh_bid)
    - ask: create a limit order to sell # stocks at some price 
      - stored self.asks as (price, #)
      - lowest ask tracked as (self.low_ask, self.nlow_ask)
    also stores an evolving self.midprice self.midprice, self.spread, self.delta_b, self.delta_a
    """
    def __init__(self, baseline=100, n_to_add=100):
        # keep track of limit orders
        self.bids = []
        self.asks = []
        # and their relevant pricing dynamics
        self.midprice = 0
        self.spread   = 0
        self.delta_b  = 0
        self.delta_a  = 0

        # BROWNIAN MIDPRICE
        self.drift = 3.59e-6
        self.scale = 2.4e-3
        self.max_t = 1
        self.baseline = baseline
        self.midprice = self.baseline
        self.model = BrownianMotion(drift=self.drift, scale=self.scale, t=self.max_t)

        # ensure there are ALWAYS enough stocks to buy/sell stuff
        self.worst_bid = 0.01
        self.worst_ask = 100*baseline
        self.n_to_add = n_to_add
    
    def copy(self):
        """ Create a copy of the order book """
        new_book = OrderBook(self.baseline, self.n_to_add)
        new_book.bids = self.bids.copy()
        new_book.asks = self.asks.copy()
        new_book.midprice = self.midprice
        new_book.spread = self.spread
        new_book.delta_b = self.delta_b
        new_book.delta_a = self.delta_a
        return new_book
    
    def is_empty(self):
        """ Check if there are either no bids or no asks """
        return not (len(self.bids) and len(self.asks))
    
    def update_midprice(self):
        """ Update midprice of market """
        self.midprice += self.model._sample_brownian_motion(1)[1]
        self.recalculate()

    def recalculate(self):
        """ Recalculate self.midprice and self.spread """
        self.spread = 0
        self.delta_b = self.delta_a = 0
        self.low_ask = self.high_bid = 0
        self.nlow_ask = self.nhigh_bid = 0
        if len(self.asks):
            self.low_ask, self.nlow_ask = self.asks[0]
            if len(self.bids):
                self.high_bid = -self.bids[0][0]; self.nhigh_bid = self.bids[0][1]
                # symmetric midprice
                # self.midprice = (self.low_ask + self.high_bid)/2
# MAKE SURE DELTA_A AND DELTA_B ARE POSITIVE
                while self.high_bid > self.midprice:
                    price, n = heapq.heappop(self.bids)
                    if len(self.bids) == 0:   # ran out of bids??
                        self.bid(self.n_to_add, round(self.midprice,2) - 0.01)
                    self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]
                while self.low_ask < self.midprice:
                    price, n = heapq.heappop(self.asks)
                    if len(self.asks) == 0:  # ran out of asks??
                        self.ask(self.n_to_add, round(self.midprice,2) + 0.01)
                    self.low_ask, self.nlow_ask = self.asks[0]
                self.spread = self.low_ask - self.high_bid
                self.delta_b = self.midprice - self.high_bid
                self.delta_a = self.low_ask - self.midprice
                if self.spread < 0:
                    print("ERROR: unrealistic spread!!")
        elif len(self.bids):
            self.high_bid, self.nhigh_bid = -self.bids[0][0], self.bids[0][1]

    def buy(self, volume: int, maxprice=0.0):
        """ Buy (up to) volume stocks to lowest-priced limit-sell (ask) orders
        returns tuple of int: num stocks bought,
                         list of (price, n_stocks) orders that have been bought
        optionally, only buy stocks valued below max price"""
        n_bought = total_bought = 0; do_update = False
        # bought = []
        while volume > 0 and len(self.asks):
            if maxprice:  # only buy stocks less than max price
                if self.asks[0][0] > maxprice: break
            # buy up to volume stocks from the n available at this price
            price, n = heapq.heappop(self.asks)
            n_bought_rn = min(n, volume)
            n_bought += n_bought_rn
            #bought.append((price, n_bought_rn))
            total_bought += price*n_bought_rn
            # place remaining portion of limit order back
            if volume < n: 
                heapq.heappush(self.asks, (price, n-volume))
            else: 
                do_update = True
            volume -= n
        # disencentivize running out of asks - you paid someone to buy your stock
        if volume > 0 and not len(self.asks):
            total_bought = -1
            n_bought     = 1
        if do_update: self.recalculate()
        return n_bought, total_bought

    def sell(self, volume: int, minprice=0.0):
        """Sell (up to) volume stocks to highest-priced limit-buy (bid) orders
        optionally, only sell stocks valued above min price"""
        n_sold = total_sold = 0; do_update = False
        #sold = []  # keep track of orders sold
        while volume > 0 and len(self.bids):
            if minprice:  # only sell stocks greater than min price
                if -self.bids[0][0] < minprice: break
            price, n = heapq.heappop(self.bids)
            price = -price  # convert back to normal price
            # sell up to volume stocks to the n available at this price
            n_sold_rn = min(n, volume)
            n_sold += n_sold_rn
            #sold.append((price,n_sold_rn))
            total_sold += price*n_sold_rn
            # place remaining portion of limit order back
            if volume < n: 
                heapq.heappush(self.bids,(-price, n-volume))
            else: 
                do_update = True
            volume -= n
        # disencentivize running out of bids - you paid someone and gave them your stock
        if volume > 0 and not len(self.bids):
            total_sold = -1
            n_sold     = 1
        if do_update: self.recalculate()
        return n_sold, total_sold

    def bid(self, volume: int, price: float):
        """ Add a limit-buy order. Sorted highest-to-lowest """
        price = round(price, 2)  # can only buy/sell in cents
        if volume == 0:
            return
        # ALLOW AGENT TO WIDEN THE SPREAD by eating the book
        if volume < 0:
            nsold, sold = self.sell(-volume, minprice=price)
            self.recalculate()
            return
        # buying higher than lowest sell -> market buy instead
        if len(self.asks):
            if price >= self.asks[0][0]:
                nbought, bought = self.buy(volume, maxprice=price)
                volume -= nbought
                if volume == 0: return  # all eaten!
        heapq.heappush(self.bids, (-price, volume))
        # is now highest buy order, recalculate
        if price == -self.bids[0][0]:
            self.recalculate()

    def ask(self, volume: int, price: float):
        """ Add a limit-sell order """
        price = round(price, 2)  # can only buy/sell in cents
        if volume == 0:
            return
        # ALLOW AGENT TO WIDEN THE SPREAD by eating the book
        if volume < 0:
            nbought, bought = self.buy(-volume, maxprice=price)
            self.recalculate()
            return
        # selling lower than highest buy order -> sell some now!
        if len(self.bids):
            if price <= -self.bids[0][0]:
                nsold, sold = self.sell(volume, minprice=price)
                volume -= nsold
                if volume == 0: return  # all eaten!
        heapq.heappush(self.asks, (price, volume))
        # is now lowest sell order, recalculate
        if price == self.asks[0][0]:
            self.recalculate()
    
    def plot(self, wait_time=0.2, title="Order Book", market_order=False, limit_order=False):
        """Make histogram of current limit-orders"""
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set(xlabel='Price per Share',ylabel='Volume')
        # normalize the bin widths
        nbins = 100  # total nbins across range of data
        self.recalculate()
        if len(self.asks) and len(self.bids):
            highest_ask = max([a[0] for a in self.asks])
            lowest_bid  = min([-b[0] for b in self.bids])
            pricerange = highest_ask - lowest_bid
            brange = max(self.high_bid - lowest_bid,0.01)
            arange = max(highest_ask - self.low_ask,0.01)
            bbins = math.ceil(nbins*brange/pricerange)
            abins = math.ceil(nbins*arange/pricerange)
            # plot the bids and asks
            ax.hist([-b[0] for b in self.bids], weights = [b[1] for b in self.bids], bins=bbins,label='Bids',edgecolor='black',linewidth=0.5)
            ax.hist([s[0] for s in self.asks], weights = [s[1] for s in self.asks], bins=abins,label='Asks',edgecolor='black',linewidth=0.5)
        else:  # just bids or just asks
            if len(self.asks):
                highest_ask = max([a[0] for a in self.asks])
                arange = max(highest_ask - self.low_ask,0.01)
                abins = nbins
                ax.hist([s[0] for s in self.asks], weights = [s[1] for s in self.asks], bins=abins,label='Asks',edgecolor='black',linewidth=0.5)
            elif len(self.bids):
                lowest_bid  = min([-b[0] for b in self.bids])
                brange = max(self.high_bid - lowest_bid,0.01)
                bbins = nbins
                ax.hist([-b[0] for b in self.bids], weights = [b[1] for b in self.bids], bins=bbins,label='Bids',edgecolor='black',linewidth=0.5)
        # set plotting stuff
        if self.midprice:  # add dashed line for midprice
            lowy, highy = ax.get_ylim()
            ax.plot([self.midprice,self.midprice],[lowy,highy*0.8],linestyle='dashed',color='black')
            if not market_order:
                title += f" [{round(self.midprice,2)}, ({uFormat(self.delta_b,0)}, {uFormat(self.delta_a,0)})]"
        if not isinstance(market_order, bool):  # plot market orders
            # market_order = (-n_bid, bid, -n_ask, ask)
            if market_order[0]:
                width = brange/bbins
                ax.hist([market_order[1]], weights = [-market_order[0]], bins = [market_order[1]-width/2,market_order[1]+width/2], edgecolor='black', linewidth=1, color=f'C{3}')  # purple for bid-hitting
            if market_order[2]:
                width = arange/abins
                ax.hist([market_order[3]], weights = [-market_order[2]], bins = [market_order[3]-width/2,market_order[3]+width/2], edgecolor='black', linewidth=1, color=f'C{2}')  # orange for ask-lifting
        if not isinstance(limit_order, bool):  # agent action
            # limit_order = (n_bid, bid, n_ask, ask)
            if limit_order[0]:
                width = brange/bbins
                ax.hist([limit_order[1]], weights = [limit_order[0]], bins = [market_order[1]-width/2,market_order[1]+width/2], edgecolor='black', linewidth=1, color=f'C{3}')
            if limit_order[2]:
                width = arange/abins
                ax.hist([limit_order[3]], weights = [limit_order[2]], bins = [market_order[3]-width/2,market_order[3]+width/2], edgecolor='black', linewidth=1, color=f'C{2}')
        plt.title(title)
        plt.legend(loc='upper center',ncol=2)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()

# we getting Pythonic up in this
if __name__ == "__main__":
    # test orderbook eating properties
    # also book.plot()
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
            book.update_midprice()
        book.plot()