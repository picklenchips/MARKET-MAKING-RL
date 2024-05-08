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
'frozen inventory' strategy
inactive trader
no limit orders and only holds inventory of q stocks until T
"""

def frozen_value(initial_wealth, stock_val, nstocks, time, 
                 gamma, sigma, terminal_time):
    first = -np.exp(-gamma*(initial_wealth+nstocks*stock_val))
    second = np.exp((gamma*nstocks*sigma)**2 * (terminal_time - time) / 2)
    return first * second