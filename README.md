# MARKET-MAKING-RL
Using reinforcement learning to make markets in the high-frequency trading setting.

1. Use high-frequency trading data and market assumptions to create an accurate model of how stock pricing dynamics change over time as a function of mid-price, spread, number of limit and market orders, etc.
2. Train an agent to learn the best way to place market and limit orders in the market in accordance with optimizing a utility function.

We begin with the simplest assumptions for market dynamics and a utility function, beginning with the theoretical framework outlined in the seminal 2006 paper "[High-frequency trading in a limit order book](https://math.nyu.edu/~avellane/HighFrequencyTrading.pdf)" and moving to more complicated market and utility function models. We quickly realized that this model makes too many assumptions and pivoted into our own environment.

### Organization

`main.py` is run with arguments to train a `MarketMaker` within the directory setup, storing results in `results/`. 

#### `config.py`
Hyperparameters are passed around using a `Config` instance that is used to initialize `MarketMaker`, `Policy`, `Market`, and `OrderBook` instances. 
- default hyperparameters are stored in the kwargs for `Config.__init__()`
- `get_config(args)` is used to initialize a `Config` from the arguments passed through to `main.py`
- as epochs are updated, `Config.set_name()` is used to rename all run files with the correct current epoch


$$e^{asdfasdf}$$

TODO: make this for instructions LOL

