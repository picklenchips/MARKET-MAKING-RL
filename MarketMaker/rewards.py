try:
    from MarketMaker.util import np, np2torch
    from MarketMaker.market import BaseMarket
    from MarketMaker.config import Config
except ModuleNotFoundError:
    from util import np, np2torch
    from market import BaseMarket
    from config import Config

class Market(BaseMarket):
    def __init__(self, inventory: int, wealth: float, config: Config):
        super().__init__(inventory, wealth, config)

    def reward(self, r_state):
        """ 
        set the immediate reward for the agent every dt
        If we do just dW here, the sum of these actions should give
            us the best dW. 
        However, all of the intermediate actions will normally have 
            this base weighting...
            which would generally discentivize market_selling to the 
            highest bid and market_buying to the lowest ask 
        In other words, our agent would learn to only add asks, no  
            bids, as bids give you a positive dI but a negative 
            dW. 
            This was why the first naive algorithm that weighted 
            the sign of dI by the negative exponential for
            the time left, 
            we were able to posibtively factor in a change in dI that 
            would then add value. 
        If we really wanted to, we could just 
            turn this in to an intermediate reward function fully, where we just learn the function, 
            - dW + dI*self.book.midprice
        Then the inetermediate reward is really just a baby version 
            of the true reward and the final reward. However, should 
            we then doubly inforce the final reward as W? 
        """
        # dW, dI, time_left = reward_state
        if not self.config.immediate_reward: 
            return 0
        if isinstance(r_state, tuple):
            r_state = np.array(r_state)
        reward = self.a*r_state[...,0]
        if self.config.add_inventory:
            reward += np.exp(-self.b*r_state[...,2]) * np.sign(r_state[...,1])
        if self.config.add_time:
            reward += self.c*(self.max_t - r_state[...,2])
        return reward
        
    def final_reward(self, wealth, inventory, midprice):
        return wealth+inventory*midprice
        
    # --- OLD STUFF --- #
    def avellaneda_lambda_buy(self, delta_a):
        k = self.alpha * self.K_b
        A = self.Lambda_b / self.alpha
        return A * np.exp(-k*delta_a)

    def avellaneda_lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * np.exp(-k*delta_b)

    def frozen_reward(self, initial_wealth, stock_val, nstocks, time_left):
        first = -np.exp(-self.gamma*(initial_wealth + nstocks*stock_val))
        second = np.exp((self.gamma*nstocks*self.sigma)**2 * time_left / 2)
        return first * second

    def reservation_price(self, midprice, inventory, t_left):  # reservation / indifference price
        return midprice - inventory * self.gamma * self.sigma**2 * t_left

    def optimal_spread(self, time_left):
        return self.gamma * self.sigma**2 * time_left + 2*np.log(1+2*self.gamma/(self.alpha*(self.K_b+self.K_s)))/self.gamma