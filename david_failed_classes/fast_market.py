from MarketMaker.Market.limit_order_book import OrderBook
from MarketMaker.util import uFormat, mpl, plt, np, np2torch, build_mlp
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy import stats
from glob import glob
import argparse, os

SAVEDICT = os.getcwd() + '/trained_models/'

class MarketMaker():
    def __init__(self, inventory, wealth, dt=1e-3, 
                 gamma=1, sigma=1e-2, terminal_time=1,
                 discount=0.99):
        self.I = inventory
        self.W = wealth
        self.book = OrderBook()
        self.alpha = 1.53
        self.Lambda_b = 20
        self.Lambda_s = 20  
        self.K_b = 1
        self.K_s = 1
        self.sigma = sigma
        self.gamma = gamma 
        self.terminal_time = terminal_time
        self.dt = dt
        self.a = 1
        self.b = 1
        self.discount = discount

    def initialize_book(self, mid=533, spread=10, nstocks=100, nsteps=100, substeps=1):
        if self.book: del(self.book)
        self.book = OrderBook(mid)
        self.book.bid(nstocks // 2, mid - spread / 2)
        self.book.ask(nstocks // 2, mid + spread / 2)
        for t in range(nsteps):
            dW, dI = self.market_step(substeps)
            state = self.observe_state()
            self.act(state)
    
    def initialize_networks(self, obs_dim=5, act_dim=4, value_dim=11, hidden_dim=10, lr=1e-3):
        nlayers = 2
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.val_dim = value_dim  
        self.policy = build_mlp(obs_dim, act_dim, nlayers, hidden_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr)
        self.value = build_mlp(value_dim, 1, nlayers, hidden_dim)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr)
    
    def lambda_buy(self, delta_a):
        k = self.alpha * self.K_b
        A = self.Lambda_b / self.alpha
        return A * torch.exp(-k * delta_a)

    def lambda_sell(self, delta_b):
        k = self.alpha * self.K_s
        A = self.Lambda_s / self.alpha
        return A * torch.exp(-k * delta_b)
    
    def market_step(self, nsteps=1):
        dW = torch.tensor(0.0, device='cuda')
        dI = torch.tensor(0.0, device='cuda')
        for step in range(nsteps):
            self.book.update_midprice()
            delta_b = torch.tensor(self.book.delta_b, device='cuda')
            delta_a = torch.tensor(self.book.delta_a, device='cuda')
            nbuy = torch.poisson(self.lambda_buy(delta_a))
            nsell = torch.poisson(self.lambda_sell(delta_b))
            n_ask_lift, bought = self.book.buy(nbuy.item())
            n_bid_hit, sold = self.book.sell(nsell.item())
            dW += bought - sold
            dI += n_bid_hit - n_ask_lift
            # Money made from market orders is good but
            # didn't Ryan say dW looks like respective inventories
            # times best bids, asks?
        return dW, dI

    def observe_state(self):
        return (
            torch.tensor(self.book.nhigh_bid, device='cuda'), 
            torch.tensor(self.book.high_bid, device='cuda'), 
            torch.tensor(self.book.nlow_ask, device='cuda'), 
            torch.tensor(self.book.low_ask, device='cuda')
        )

    def act(self, state: tuple):
        """ Perform action on order book (dim=4 or dim=5)
        - Inputs: (n_bid, bid_price, n_ask, ask_price) or (n_bid, bid_price, n_ask, ask_price, time_left)
        - action is only limit orders (for now) 
            - can be extended to (n_bid, bid_price, n_ask, ask_price, n_buy, n_sell)
        - returns action taken, (n_bid, bid_price, n_ask, ask_price)
        - naive sets price variance of 'stupid' policy taken around midprice
        """
        if len(state) == 5:  # use NN policy
            n_bid, bid_price, n_ask, ask_price, time_left = state
            state_tensor = torch.tensor([n_bid, bid_price, n_ask, ask_price, time_left], device='cuda')
            action = self.policy(state_tensor).detach()
            n_bid, bid_price, n_ask, ask_price = action

        elif len(state) == 4:  # default "naive" policy
            n_bid, bid_price, n_ask, ask_price = state
            delta_b = torch.tensor(self.book.delta_b, device='cuda')
            delta_a = torch.tensor(self.book.delta_a, device='cuda')
            bid_price = torch.normal(bid_price, delta_b / 4)
            ask_price = torch.normal(ask_price, delta_a / 4)
            n_bid = torch.poisson(n_bid / 2)
            n_ask = torch.poisson(n_ask / 2)
            
        if torch.isnan(n_bid):
            n_bid = torch.tensor(0, device='cuda')
        if torch.isnan(bid_price):
            bid_price = torch.tensor(0.0, device='cuda')
        if torch.isnan(n_ask):
            n_ask = torch.tensor(0, device='cuda')
        if torch.isnan(ask_price):
            ask_price = torch.tensor(0.0, device='cuda')

        self.limit_act(n_bid, bid_price, n_ask, ask_price)
        return (n_bid, bid_price, n_ask, ask_price)


    def limit_act(self, n_bid, bid_price, n_ask, ask_price):
        """ Perform a limit order action, making (up to) 2 limit orders: a bid and ask
        - if n_bid or n_ask < 0.5, will not bid or ask """
        # make sure that bid price is always less than ask price
        bid_price = torch.clip(bid_price, 0, torch.tensor(self.book.low_ask, device='cuda'))
        ask_price = torch.clip(ask_price, torch.tensor(self.book.high_bid, device='cuda'), float('inf'))
        # can only bid integer multiples
        n_bid = round(n_bid.item())
        n_ask = round(n_ask.item())
        self.book.bid(n_bid, bid_price.item())
        self.book.ask(n_ask, ask_price.item())
        return n_bid, bid_price, n_ask, ask_price

    def frozen_reward(self, initial_wealth, stock_val, nstocks, time_left):
        first = -torch.exp(-self.gamma * (initial_wealth + nstocks * stock_val))
        second = torch.exp((self.gamma * nstocks * self.sigma)**2 * time_left / 2)
        return first * second

    def immediate_reward(self, r_state):
        if isinstance(r_state, tuple):
            r_state = torch.stack(r_state)
        return self.a * r_state[..., 0] + torch.exp(-self.b * r_state[..., 2]) * torch.sign(r_state[..., 1])
    
    def final_reward(self, dW, inventory, midprice):
        return dW + inventory * midprice

    def reservation_price(self, midprice, inventory, t_left):
        return midprice - inventory * self.gamma * self.sigma**2 * t_left

    def optimal_spread(self, time_left):
        return self.gamma * self.sigma**2 * time_left + 2 * torch.log(1 + 2 * self.gamma / (self.alpha * (self.K_b + self.K_a))) / self.gamma
    
    def simulate(self, nbatch=1000, track_all=False, action=''):
        """ Naively iterate through order book to show evolution over time using current policy algorithm 
        Sample a batch of trajectories under some policy
        Inputs:
        - nbatch = number of trajectories to sample
        - track_all = keep track of wealth, inventory, midprices over time
        - action = 'naive' or 'avellaneda', or anything else
        Output: (trajectories, rewards)
        - trajectories = (nbatch x num_times x val_dim) np.nd.arary or tensor
        - rewards = (nbatch x num_times) np.ndarray or tensor
        if track_all, output: (trajectories, rewards, wealth, inventory, midprices) """
        T = self.terminal_time
        dt = self.dt
        nt = int(T / dt) + 1
        trajectories = torch.zeros((nbatch, nt, self.val_dim), device='cuda')
        rewards = torch.zeros((nbatch, nt), device='cuda')
        if track_all:
            wealth = torch.empty((nbatch, nt), device='cuda')
            inventory = torch.empty((nbatch, nt), dtype=torch.long, device='cuda')
            midprices = torch.empty((nbatch, nt), device='cuda')
        with tqdm(total=nbatch) as pbar:
            pbar.set_description("Creating Batch...")
            for b in range(nbatch):
                self.initialize_book()
                W = torch.tensor(float(self.W), device='cuda')
                I = torch.tensor(self.I, dtype=torch.long, device='cuda')
                if track_all:
                    wealth[b, 0] = W
                    inventory[b, 0] = I
                    midprices[b, 0] = self.book.midprice
                for t in range(nt - 1):
                    time_left = torch.tensor(self.terminal_time - t * self.dt, device='cuda')
                    state = self.observe_state()
                    if action == 'avellaneda':
                        action = self.act((W, I, time_left))
                    elif action == 'naive':
                        action = self.act(state)
                    else:
                        action = self.act(state + (time_left,))
                    dW, dI = self.market_step()
                    W += dW.float()
                    I += dI.long()
                    # if dI > 10:
                    #     print(f"Inventory: {I}, Wealth: {W}")
                    if track_all:
                        wealth[b, t + 1] = wealth[b, t] + dW.float()
                        inventory[b, t + 1] = inventory[b, t] + dI.long()
                        midprices[b, t + 1] = self.book.midprice
                    reward_state = (dW, dI, time_left)
                    rewards[b, t] = self.immediate_reward(reward_state)
                    state_tensor = torch.tensor(state, device='cuda') if isinstance(state, tuple) else state
                    action_tensor = torch.tensor(action, device='cuda') if isinstance(action, tuple) else action
                    reward_state_tensor = torch.tensor(reward_state, device='cuda') if isinstance(reward_state, tuple) else reward_state
                    trajectories[b, t] = torch.cat((state_tensor, action_tensor, reward_state_tensor))
                rewards[b, t] = self.final_reward(dW.float(), inventory[b, -1], midprices[b, -1])
                pbar.update(1)
                pbar.set_postfix_str(f"Reward {rewards[b, -1]}", refresh=True)
        if track_all:
            return trajectories, rewards, wealth, inventory, midprices
        return trajectories, rewards

    def get_returns(self, rewards):
        """ Compute returns from batched rewards using cumulative sum
        Inputs:
        - rewards (nbatch x nt) tensor
        """
        nbatch, nt = rewards.shape
        returns = torch.zeros_like(rewards, device='cuda')
        for b in range(nbatch):
            cumulative_return = 0
            for t in reversed(range(nt)):
                cumulative_return = rewards[b, t] + self.discount * cumulative_return
                returns[b, t] = cumulative_return
        return returns
    
    def plot(self, wealth, inventory, midprices, title=''):
        """ Plot data from a batch of trajectories
        Inputs: (nbatch x nt) tensors """
        times = torch.arange(0, self.terminal_time + self.dt, self.dt, device='cuda')
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        for i, y, name in zip((0, 1, 2), (wealth, inventory, midprices), ('Wealth', 'Inventory', 'Midprice')):
            ax = axs[i]
            ax.set(ylabel=name)
            y = y.float()  # Convert y to floating point type
            ys = torch.mean(y, axis=0)
            yerrs = torch.std(y, axis=0) / torch.sqrt(torch.tensor(y.shape[0], dtype=torch.float32, device='cuda'))
            ax.fill_between(times.cpu().numpy(), (ys - yerrs).cpu().numpy(), (ys + yerrs).cpu().numpy(), alpha=0.25, color=f"C{i}")
            ax.plot(times.cpu().numpy(), ys.cpu().numpy(), color=f"C{i}")
        axs[2].set(xlabel="Time")
        i += 1
        y = wealth + inventory * midprices
        y = y.float()  # Convert y to floating point type
        ys = torch.mean(y, axis=0)
        yerrs = torch.std(y, axis=0) / torch.sqrt(torch.tensor(y.shape[0], dtype=torch.float32, device='cuda'))
        axs[0].fill_between(times.cpu().numpy(), (ys - yerrs).cpu().numpy(), (ys + yerrs).cpu().numpy(), alpha=0.25, color=f"C{i}")
        axs[0].plot(times.cpu().numpy(), ys.cpu().numpy(), label='Total Value', color=f"C{i}")
        axs[0].legend()
        if title: axs[0].set_title(title)
        plt.show()

    def get_advantages(self, trajectories, returns):
        """ Compute advantages from batched rewards and batched trajectories 
        - returns (nbatch x nt) tensor
        - trajectories (nbatch x nt x val_dim) tensor
        - run value network on trajectories to get value estimates """
        trajectories = np2torch(trajectories)
        values = self.value(trajectories).squeeze().detach().cpu().numpy()
        returns = returns.cpu().numpy()
        advantages = returns - values
        return (advantages - advantages.mean()) / (advantages.std())

    def update_value(self, trajectories, rewards):
        """ use MSE loss to train value function with TD(lambda) """
        trajectories = np2torch(trajectories)
        with torch.no_grad():
            values = self.value(trajectories).squeeze()
        td_lambda_targets = self.compute_td_lambda_targets(rewards, values)
        act_val = td_lambda_targets.clone().detach().to('cuda')
        pred_val = self.value(trajectories).squeeze()
        loss = torch.mean((act_val - pred_val) ** 2)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def update_policy(self, trajectories, advantages):
        trajectories = np2torch(trajectories)
        advantages = torch.tensor(advantages, device='cuda')
        states = trajectories[..., [0, 1, 2, 3, -1]]
        actions = self.policy(states)
        loss = torch.log(advantages[..., None] * torch.log(actions))
        loss = loss.mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def save(self, ne, nb, nt, name='MC'):
        name = f"{SAVEDICT}{name}_{ne}_{nb}_{nt}_"
        torch.save(self.value.state_dict(), name + "val.pth")
        print(f"Saved value network to {name}val.pth")
        torch.save(self.policy.state_dict(), name + "pol.pth")
        print(f"Saved policy network to {name}pol.pth")

    def load(self, ne, nb, nt, name='MC'):
        name = f"{SAVEDICT}{name}_{ne}_{nb}_{nt}_"
        self.initialize_networks()
        self.value.load_state_dict(torch.load(name + "val.pth"))
        self.policy.load_state_dict(torch.load(name + "pol.pth"))
        print(f"Loaded networks from {name}(val,pol).pth")

    def compute_td_lambda_targets(self, rewards, values, lambda_=0.95):
        nbatch, nt = rewards.shape
        td_lambda_targets = torch.zeros_like(rewards, device='cuda')
        for b in range(nbatch):
            g_t_lambda = torch.tensor(0.0, device='cuda')
            for t in reversed(range(nt)):
                g_t_lambda = rewards[b, t] + self.discount * ((1 - lambda_) * values[b, t] + lambda_ * g_t_lambda)
                td_lambda_targets[b, t] = g_t_lambda
        return td_lambda_targets

def train_market(num_epochs=100, batch_size=1000, timesteps=5000, plot_after=1000):
    obs_dim = 4; act_dim = 4
    rew_dim = 2
    value_dim = obs_dim + act_dim + rew_dim + 1
    policy_update = 1

    dt = 0.005
    nt = timesteps 
    terminal_time = nt * dt

    mm = MarketMaker(0, 0, dt=dt, gamma=1, sigma=1, terminal_time=terminal_time)
    mm.initialize_networks(value_dim=value_dim)
    
    with tqdm(total=num_epochs) as pbar:
        pbar.set_description("Training Market Maker...")
        for epoch in range(num_epochs):
            trajectories, rewards, wealth, inventory, midprice = mm.simulate(nbatch=batch_size, track_all=True)
            if (epoch + 1) % plot_after == 0:
                mm.plot(wealth, inventory, midprice, title=f'epoch {epoch}')
            returns = mm.get_returns(rewards)
            advantages = mm.get_advantages(trajectories, returns)
            mm.update_value(trajectories, rewards)
            if (epoch + 1) % policy_update == 0:
                mm.update_policy(trajectories, advantages)
            pbar.update(1)
            pbar.set_postfix_str(f"Reward {rewards[:, -1].mean()}", refresh=True)
    mm.save(num_epochs, batch_size, timesteps)

parser = argparse.ArgumentParser()
parser.add_argument("-ne", "-n_epochs", dest='ne', type=int, default=1)
parser.add_argument("-nb", "-n_batches", dest='nb', type=int, default=10)
parser.add_argument("-nt", "-n_times", dest='nt', type=int, default=100)
parser.add_argument("-test_initialization", dest='testinitial', default=False, action='store_true')
parser.add_argument("-l", "--load", nargs='+', default=[])

if __name__ == "__main__":
    args = parser.parse_args()
    if len(args.load) >= 3:
        load = [int(i) for i in args.load[:3]]
        if len(args.load) > 3:
            load.append(args.load[3])
        nb = load[1]
        nt = load[2]
        dt = 0.001
        mm = MarketMaker(0, 0, dt=dt, terminal_time=nt * dt)
        mm.load(*load)
        t, r, w, i, m = mm.simulate(nb, track_all=True)
        mm.plot(w, i, m, title=f'Loaded Model: {", ".join(args.load)}')
    if args.testinitial:
        mm = MarketMaker(0, 0)
        while input("Continue? (y/n): ").strip().lower() == 'y':
            mm.initialize_book(nsteps=100)
            mm.book.plot()
    elif not args.load:
        train_market(args.ne, args.nb, args.nt)
