import numpy as np

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




