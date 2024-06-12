# MARKET-MAKING-RL
Creating a market-making RL agent in the high-frequency trading (HFT) setting for Stanford's CS234 final project. 
Our full paper is found [here](https://github.com/user-attachments/files/15811478/CS234_Paper___HFT_Market_Making.3.pdf). 
As opposed to past RL attempts on the problem of market-making in the HFT setting, we physically implement the limit order book (LOB) when simulating our algorithm, which means that the market dynamics are explieiclty dependent on the (sum) of the market-makers actions. 

## Model & Assumptions

Market-makers provide the liquidity of the stock market, allowing people to buy and sell stocks. 
Liquidity is provided as "limit orders" in the form of **asks** that allow the market to buy stocks and **bids** that allow the market to sell stocks.
A limit order has both a price and a number of stocks, and a **limit order book (LOB)** contains all bids and asks provided by a market maker.
As a market-maker is only one provider of liquidity among many for stock, the **midprice** of a stock is decoupled from any one market and evolves over time according to a stochastic Brownian motion. The **bids** and **asks** are centered around the midprice with a certain spread of $\delta_b$, $\delta_a$ respectively:

<img width=80% alt="limit-order book" margin-left=auto margin-righ=auto
  src="https://github.com/picklenchips/MARKET-MAKING-RL/assets/77514590/5ee3a6f3-f357-4c97-833e-840bc96d7b17">

The market only interacts with the bids and asks closest to the midprice, buying the cheapest ask and selling the highest bid. 
Modeling the rate of market orders is an active research field but generally follows a Poisson process with a rate $\lambda(q,\delta)$ where $q$ is the quantity of the best bid/ask and $\delta$ is the spread of the best bid/ask from the midprice. 
We begin with a base high-frequency trading market rates that are only a function of $\delta$ derived in 2006 by [Avellaneda & Stoikov](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf), then switched to a function of only $q$ derived in 2009 by [Mykland & Zhang](https://doi.org/10.3982/ECTA7417), and finally found a function of both derived in 2017 by [Toke & Yoshida](https://doi.org/10.1080/14697688.2016.1236210).
We synthesize these functions to find the most robust and realistic model, 

$$\lambda(s,\delta) = \exp\left( \vec{\beta} \cdot 
\left[\ln(1 + q),\ (\ln(1 + q))^2,\ \ln(1 + \delta),\ (\ln(1 + \delta))^2,\ \ln(1+\delta+q) \right] \right),$$

where $\vec{\beta}$ contains the 6 parameters $\beta_i$ which we fit by fitting a stochastic linear regression on a day's worth of S&P500 data. We verify that the shape of the parameters match our expectation by plotting the derived lambda equation (as a function of $s$, $\delta$ separately) [on Desmos](https://www.desmos.com/calculator/kgcnsxxsdw).



## Organization

`main.py` is run with arguments to train a `MarketMaker` within the directory setup, storing results in `results/`. The path of the code is as follows:
1. a `Config` is created from the arguments to store all hyperparameters and flags as well as organize the files in `results/`
2. a `MarketMaker` is created from the `Config`
3. `MarketMaker` creates a `Policy` and `Market`, loading an existing `BasePolicy` and `BaseNetwork` state dict if necessary
4. `Market` creates an `OrderBook` upon which to simulate trajectories
5. The `MarketMaker` is then either trained or plotted, according to args.

Arguments to Control Behavior:
- `--expand` number of epochs to expand a finished run to
- `--load` filename or path to load previous model with
- `--plot` just plot a model
- `--good-results` plot all models located in the directory `/good-results`
- `--plot-after`, `-pa` number of epochs after which to plot a full batch
Arguments to Control Hyperparameters:
- Training
  - `-nt` number of timesteps per trajectory
  - `-nb` number of trajectories to sample per epoch
  - `-ne` number of epochs to train policy on
  - `-lr` float learning rate to pass to Adam optimizer
  - `--update-freq` number of gradient steps to perform each epoch, default 5
  - `--uniform` use a uniform trajectory length where when the LOB dies, all observations are zero
- Policy: default is MC returns, PPO policy update, use advantages
  - `--td` use td-eligibility-trace returns instead of the usual monte-carlo discounted returns
  - `--noppo`, `-np` use REINFORCE istead of the default PPO
  - `--noclip`, `-nc` when using PPO, don't clip the policy ratio
  - `--noadv`, `-na` use the returns instead of advantages
- Reward: default is an immediate dW reward
  - `--no-immediate`, `-ni` dont add any intermediate rewards
  - `--always-final`, `-af` always add a final reward
  - `--add-time`, `-at` add a time reward
  - `--add-inventory`, `-ai` add an inventory reward
- Initial State
  - `--book-size`, `-bs` initial number of stocks in the order book, default 10000

#### `config.py`

Hyperparameters are passed around using a `Config` instance that is used to initialize `MarketMaker`, `Policy`, `Market`, and `OrderBook` instances. 
- default hyperparameters are stored in the kwargs for `Config.__init__()`
- `get_config(args)` is used to initialize a `Config` from the arguments passed through to `main.py`
- as epochs are updated, `Config.set_name()` is used to rename all run files with the correct current epoch


## TODO: 
- implement more efficient gradients with sparse CSR or CSC arrays (`MaskedMarketMaker`) for faster running
- incorporate testing of more reward functions, such as the CARA final utility from [Lim & Grose, 2018](https://discovery.ucl.ac.uk/id/eprint/10116730/1/RLforHFMM.pdf) and the various reward schema used in [Guo, Lin, and Huang, 2023](https://arxiv.org/abs/2305.15821).
- parameter tune TD $\lambda$ eligibility traces
- implement dynamically taking in the $p$ past trajectories and using more layers to generate a more robust policy
- create full CUDA version that uses tensor operations whenever possible, so that larger batch numbers can be used for better, more robust policies
  - create full MPS version (M1), which should just be changing the `.to(device)` line
- actually find the correct parameters for the market transaction rate instead of just eyeballing from Desmos


