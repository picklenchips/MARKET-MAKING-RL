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
objective is
$u(s, x, q, t) = \max_{\delta^a,\delta^b} 
\mathbb{E}_t \left[-\exp\left(-\gamma(X_T + q_T S_T)\right)\right]$
"""


"""
model trading intensity.
assume constant frequency f^Q(x)\alpha x^{-1-\alpha}
"""
