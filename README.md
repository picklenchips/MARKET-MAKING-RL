# MARKET-MAKING-RL
Creating a market-making RL agent in the high-frequency trading setting for Stanford's CS234 final project. 
Our full paper is found [here](https://github.com/user-attachments/files/15811478/CS234_Paper___HFT_Market_Making.3.pdf)

Market-makers provide the liquidity of the stock market, allowing people to buy and sell stocks. Liquidity is provided as limit-orders in the form of **asks** that allow the market to buy stocks and **bids** that allow the market to sell stocks. As a market-maker is only one provider of liquidity among many for stock, the **midprice** of a stock is decoupled from any one market and evolves over time according to a stochastic Brownian motion. The **bids** and **asks** are centered around the midprice with a certain spread of $\delta_b$, $\delta_a$ respectively:

<img width=80% alt="LOB (1)" src="https://github.com/picklenchips/MARKET-MAKING-RL/assets/77514590/5ee3a6f3-f357-4c97-833e-840bc96d7b17">




Assume the rate of market orders that 

1. Create an accurate model
 of how stock pricing dynamics change over time as a function of mid-price, spread, number of limit and market orders, etc. that is reflected
2. Train an agent to learn the best way to place market and limit orders in the market in accordance with optimizing a utility function.

We begin with the simplest assumptions for market dynamics and a utility function, beginning with the theoretical framework outlined in the seminal 2006 paper "[High-frequency trading in a limit order book](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)" and moving to more complicated market and utility function models. We quickly realized that this model makes too many assumptions and pivoted into our own environment.

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


