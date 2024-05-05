import numpy as np
import matplotlib.pyplot as plt

import queue  # priority queue

class OrderBook():
    def __init__(self):
        self.buys = queue.PriorityQueue()
        self.sells = queue.PriorityQueue()
        self.midprice = 0
        self.spread = 0
    
    def market_buy(self, n_stocks):
        pass

    def market_sell(self):
        pass

    def limit_buy(self):
        pass

    def limit_sell(self):
        pass

    # plot the buy, sell thingies
    def plot(self):
        pass