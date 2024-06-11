# MARKET-MAKING-RL
Using reinforcement learning to make markets in the high-frequency trading setting.

1. Use high-frequency trading data and market assumptions to create an accurate model of how stock pricing dynamics change over time as a function of mid-price, spread, number of limit and market orders, etc.
2. Train an agent to learn the best way to place market and limit orders in the market in accordance with optimizing a utility function.

We begin with the simplest assumptions for market dynamics and a utility function, beginning with the theoretical framework outlined in the seminal 2006 paper "[High-frequency trading in a limit order book](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)" and moving to more complicated market and utility function models. We quickly realized that this model makes too many assumptions and pivoted into our own environment.

### Organization

`main.py` is run with arguments to train a `MarketMaker` within the directory setup, storing results in `results/`. 

Arguments to Control Behavior:
- `--expand` number of epochs to expand a finished run to
- `--load` filename or path to load previous model with
- `--plot` just plot a model
- `--good-results` plot all models located in the directory `/good-results`
- `--plot-after`, `-pa` number of epochs after which to plot a full batch
- 
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
- implement more efficient gradients with sparse CSR or CSC arrays (`MaskedMarketMaker`)
- create full CUDA version
- create full MPS version (M1)

