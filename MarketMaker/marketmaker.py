try:
    from MarketMaker.util import np, get_logger, torch, plot_WIM, export_plot, np2torch
    from MarketMaker.config import Config
    from MarketMaker.rewards import Market
    from MarketMaker.policy import Policy
except ModuleNotFoundError:
    from util import np, get_logger, torch, plot_WIM, export_plot, np2torch
    from config import Config
    from rewards import Market
    from policy import Policy
from tqdm import tqdm
from glob import glob
import os, sys
import torch.masked as masked

class UniformMarketMaker():
    """ a market maker with uniform time steps for all trajectories """
    def __init__(self, config: Config, inventory=0, wealth=0) -> None:
        self.config = config
        self.market = Market(inventory, wealth, config)
        self.logger = self.config.logger
        self.P = Policy(config)
        self.logger.info(self.config)
        self.do_ppo = config.do_ppo
        
        msg = self.P.init_policy(self.market, new_train=False)  # train a more robust? initial policy
        self.logger.info(msg)
        self.dt = config.dt; self.max_t = config.max_t
        self.nt = config.nt
        self.final_returns = []  # track returns during training
        self.final_values  = []  # track values during training

    def save(self, epoch, final_epoch=False):
        """ Save a trained market maker model """
        old_out = self.config.out
        old_dir = self.config.save_dir
        name, out = self.config.set_name(epoch)
        if self.config.use_baseline:
            if os.path.exists(old_out+"_val.pth"): 
                os.remove(old_out+"_val.pth")
            torch.save(self.P.baseline.network.state_dict(), out+"_val.pth")
        if os.path.exists(old_out+"_pol.pth"):
            os.remove(old_out+"_pol.pth")
        torch.save(self.P.policy.state_dict(), out+"_pol.pth")
        # save final returns and final values
        if os.path.exists(old_out+"_scores.npy"): 
            os.remove(old_out+"_scores.npy")
        if os.path.exists(old_out+"_values.npy"): 
            os.remove(old_out+"_values.npy")
        finals = np.array(self.final_returns)
        values = np.array(self.final_values)
        np.save(self.config.scores_out, finals)
        np.save(self.config.values_out, values)
        # plot final returns and final values
        if os.path.exists(old_out+"_scores.png"):
            os.remove(old_out+"_scores.png")
        if os.path.exists(old_out+"_values.png"):
            os.remove(old_out+"_values.png")
        export_plot(finals,"Final Returns",self.config.name,self.config.scores_plot)
        export_plot(values,"Final Values",self.config.name,self.config.values_plot)
        msg = f'[EPOCH {epoch}] Returns {finals.shape}. Max: ({np.argmax(finals)}, {np.max(finals)}) and min: ({np.argmin(finals)}, {np.min(finals)}), values max: {np.max(values)} and min: {np.min(values)}'
        self.logger.info(msg)

    def load(self):
        """ Return a trained market maker model from same config """
        name = self.config.out
        if not os.path.exists(name+"_pol.pth"):
            raise FileNotFoundError(f"Model {name} not found")
        if self.config.use_baseline:
            self.P.baseline.network.load_state_dict(torch.load(name+"_val.pth"))
            self.logger.info(f"Loaded baseline network from {name}_val.pth")
        self.P.policy.load_state_dict(torch.load(name+"_pol.pth"))
        self.logger.info(f"Loaded policy network from {name}_pol.pth")
        self.final_returns = list(np.load(self.config.scores_out))
        self.final_values = list(np.load(self.config.values_out))
        self.logger.info(f"Loaded scores from {self.config.scores_out}")
    
    def get_paths(self, pbar, nt=None, nb=None, track_all=False):
        """ 
        get trajectories and compute rewards, only observing immediate state
        Inputs:
        - self.config.nbatch = number of trajectories to sample
        - track_all = also return (wealth, inventory, midprices)
        Output:  dictionary of all np.ndarrays (tra, obs, act, rew)
        if track_all, output: (tra, obs, act, rew, wea, inv, mid) 
        if PPO, also outputs log probs of actions in ['old'] 
        """
        dt = self.dt
        if not nt: 
            nt = self.nt
            T = self.max_t
        else:
            T = nt*dt
        nbatch = nb if nb else self.config.nb
        val_dim = self.config.val_dim
        obs_dim = self.config.obs_dim
        act_dim = self.config.act_dim
        
        trajectories = np.empty((nbatch, nt, val_dim))
        actions = np.empty((nbatch, nt, act_dim))
        rewards = np.empty((nbatch, nt))
        values  = np.empty((nbatch,))
        if self.do_ppo:
            logprobs = np.empty((nbatch, nt))
        if track_all:  # track all for later plotting?
            wealth = np.empty((nbatch, nt))
            inventory = np.empty((nbatch, nt),dtype=int)
            # track book state  (midprice, delta_b, delta_a)
            states = np.empty((nbatch, nt, 3))
        for b in range(nbatch):
            self.market.reset()
            W = self.market.W; I = self.market.I
            # timestep
            for t in range(nt):
                time_left = (T - t*dt,)
                state = self.market.state() + time_left
                if self.do_ppo:  # need to get log probability directly
                    action, logprob = self.P.policy.act(np.array(state), return_log_prob=True)
                    logprobs[b, t] = logprob
                    self.market.submit(*action)
                else:
                    action = self.market.act(state, self.P.policy.act)
                actions[b, t] = action
                dW, dI, midprice = self.market.step()   # (dW, dI, midprice)
                if dW is ValueError:
                    lambda_sell = dI; lambda_buy = midprice
                    print(f"Batch {b} step {t} INVALID lambdas, ({lambda_sell}, {lambda_buy}), on book {self.market.book}")
                    self.market.reset(plot=True, make_bell=True)
                    break
                reward_state = (dW, dI) + time_left
                rewards[b, t] = self.market.reward(reward_state)
                trajectories[b, t] = state + (dW, dI)
                W += dW; I += dI
                if track_all:
                    wealth[b, t] = W
                    inventory[b, t] = I
                    states[b, t] = (midprice, midprice-self.market.book.delta_b, midprice+self.market.book.delta_a)
            rewards[b, t] += self.market.final_reward(W, I, midprice)
            values[b] = W + I*midprice
            if pbar: pbar.update(1)
        observations = trajectories[...,:obs_dim]
        paths = {"tra": trajectories, "obs": observations, "act": actions, "rew": rewards, "val": values}
        if self.do_ppo:
            paths["old"] = logprobs
        if track_all:
            paths["wea"] = wealth
            paths["inv"] = inventory
            paths["book"] = states
        self.logger.info(f"generated trajectories of shape {paths['tra'].shape}")
        return paths

    def train(self, plot_after=False):
        """ Train number of epochs x nbatch things
        - plot_after some # of epochs to show improvement? 
        """
        nbatch = self.config.nb
        nepoch = self.config.ne - self.config.starting_epoch
        with tqdm(total=nepoch*nbatch) as pbar:  # make local pbar instance
            for epoch in range(self.config.starting_epoch, self.config.ne):
                do_plot = plot_after and ((epoch + 1) % plot_after == 0)
                pbar.set_description(f"Epoch {epoch}")
                # Get paths and returns based on trajectory type
                paths = self.get_paths(pbar, nb=nbatch, track_all=do_plot)
                self.logger.info(f"found trajectories of shape {paths['tra'].shape}")
                if self.config.trajectory == 'MC':
                    returns = self.P.get_returns(paths['rew'])
                elif self.config.trajectory == 'TD':
                    values = self.P.baseline.network(np2torch(paths['tra'])).detach().cpu().numpy()
                    returns = self.P.get_td_returns(paths['rew'], values, self.nt, nbatch)
                else:
                    raise NotImplementedError("Trajectory type not supported")
                advantages = self.P.get_advantages(returns, paths['tra'])

                # Policy updates
                for C in range(self.config.update_freq):
                    if self.config.use_baseline:
                        self.P.baseline.update_baseline(returns, paths['tra'])
                    self.P.update_policy(paths['obs'], paths['act'], advantages, old_logprobs=paths.get('old'))

                # Log the returns
                self.final_returns.append(returns[:, -1])
                self.save(epoch + 1)   # intermediately save the market maker for later loading

                # Plot if required
                if do_plot:
                    plot_WIM(paths, self.dt, title=self.config.name, savename=self.config.out+'.png')

        self.logger.info("Training complete!")  # DONZO BONZO
        self.save(epoch + 1, True)
        self.plot()
    
    def plot_book_path(self, nt=0, wait_time=1) -> None:
        """ plot the LOB for a single trajectory """
        dt = self.dt
        if not nt: 
            nt = self.nt
            T = self.max_t
        else:
            T = nt*dt
        with tqdm(total=nt) as pbar:
            pbar.set_description("Plotting Single Path...")
            self.market.reset()
            W = self.market.W; I = self.market.I
            for t in range(nt):
                time_left = (T - t*dt,)
                state = self.market.state() + time_left
                old_book = self.market.book.copy()
                # actions have been changed to deltas
                limit_act = list(self.market.act(state, self.P.policy.act))
                limit_act[1] = old_book.midprice - limit_act[1]
                limit_act[3] = old_book.midprice + limit_act[3]
                dW, dI, midprice, market_act = self.market.step(plot=True)
                title = f'{round(t*dt,3)}: ({round(W)} + {round(dW,2)}, {round(I)} + {round(dI)}, {round(W+I*old_book.midprice)} + {round(dW+dI*midprice)})'
                title = f"{', '.join(map(lambda x: str(round(x,2)), state))} -> {', '.join(map(lambda x: str(round(x,2)), limit_act))}"
                if self.book_quit and self.market.is_empty():
                    title += '\n!!! EMPTY !!!'
                old_book.plot(wait_time=wait_time, title=title, market_order=market_act, limit_order=limit_act)
                if title[-3:] == '!!!':
                    print(f'Book emptied on {t}!')
                    break
                W += dW; I += dI
                pbar.update(1)
    
    def plot(self, nb=0, nt=0, plot_book=False, wait_time=1):
        """ plot final scores and final sample path """
        nb = nb if nb else self.config.nb
        nt = nt if nt else self.config.nt
        export_plot(np.array(self.final_returns),"Final Returns",self.config.name,self.config.scores_plot)
        export_plot(np.array(self.final_values),"Final Values",self.config.name,self.config.values_plot)
        with tqdm(total=nb) as pbar:
            pbar.set_description("Plotting Final Paths...")
            paths = self.get_paths(pbar, nt=nt, nb=nb, track_all=True)
            plot_WIM(paths, self.dt, title=self.config.name, savename=self.config.wim_plot)
        # plot a single trajectory...
        if plot_book:
            self.plot_book_path(nt=nt, wait_time=wait_time)      

class MasketMarketMaker(UniformMarketMaker):
    def __init__(self, config: Config, inventory=0, wealth=0) -> None:
        super().__init__(config, inventory, wealth)
        self.book_quit = config.book_quit  # quit if book is empty

    def get_masked_paths(self, pbar, nt=0, nb=0, track_all=False) -> np.ma.MaskedArray:
        """ 
        get trajectories and compute rewards, only observing immediate state
        Inputs:
        - self.config.nbatch = number of trajectories to sample
        - track_all = also return (wealth, inventory, midprices)
        Output:  dictionary of all np.ndarrays (tra, obs, act, rew)
        if track_all, output: (tra, obs, act, rew, wea, inv, mid) 
        if PPO, also outputs log probs of actions in ['old'] 
        """
        dt = self.dt
        if not nt: 
            nt = self.nt
            T = self.max_t
        else:
            T = nt*dt
        nbatch = nb if nb else self.config.nb
        val_dim = self.config.val_dim
        obs_dim = self.config.obs_dim
        act_dim = self.config.act_dim
        
        trajectories = np.ma.empty((nbatch, nt, val_dim))
        trajectories.mask = True
        actions = np.ma.empty((nbatch, nt, act_dim))
        actions.mask = True
        rewards = np.ma.empty((nbatch, nt))
        rewards.mask = True
        if self.do_ppo:
            logprobs = np.empty((nbatch, nt))
        if track_all:  # track all for later plotting?
            wealth = np.ma.empty((nbatch, nt))
            inventory = np.ma.empty((nbatch, nt),dtype=int)
            states = np.ma.empty((nbatch, nt, 3))
        for b in range(nbatch):
            self.market.reset()
            W = self.market.W; I = self.market.I
            # timestep
            for t in range(nt):
                time_left = (T - t*dt,)
                state = self.market.state() + time_left
                if self.do_ppo:  # need to get log probability directly
                    action, logprob = self.P.policy.act(np.array(state), return_log_prob=True)
                    logprobs[b, t] = logprob
                    self.market.submit(*action)
                else:
                    action = self.market.act(state, self.P.policy.act)
                actions[b, t] = action
                dW, dI, midprice = self.market.step()   # (dW, dI, midprice)
                reward_state = (dW, dI) + time_left
                rewards[b, t] = self.market.reward(reward_state)
                trajectories[b, t] = state + (dW, dI)
                if track_all:
                    W += dW; I += dI
                    wealth[b, t] = W
                    inventory[b, t] = I
                    states[b, t] = (midprice, midprice-self.market.book.delta_b, midprice+self.market.book.delta_a)
            rewards[b, t] += self.market.final_reward(W, I, self.market.book.midprice)
            if pbar:
                pbar.update(1)
        observations = trajectories[...,:obs_dim]
        paths = {"tra": trajectories, "obs": observations, "act": actions, "rew": rewards}
        if self.do_ppo:
            paths["old"] = logprobs
        if track_all:
            paths["wea"] = wealth
            paths["inv"] = inventory
            paths["book"] = states
        return paths

    def masked_train(self, plot_after=False):
        """ train variable-length trajectories using 
        np.ma.MaskedArrays and later converting to torch.masked.MaskedTensors
        """
        nbatch = self.config.nb
        nepoch = self.config.ne - self.config.starting_epoch
        with tqdm(total=nepoch*nbatch) as pbar:  # make local pbar instance
            for epoch in range(self.config.starting_epoch, self.config.ne):
                do_plot = plot_after and ((epoch + 1) % plot_after == 0)
                pbar.set_description(f"Epoch {epoch}")
                # Get paths and returns based on trajectory type
                paths = self.get_paths(pbar, nb=nbatch, track_all=do_plot)
                if self.config.trajectory == 'MC':
                    returns = self.P.get_returns(paths['rew'])
                elif self.config.trajectory == 'TD':
                    values = self.P.baseline.network(np2torch(paths['tra'])).detach().cpu().numpy()
                    returns = self.P.get_td_returns(paths['rew'], values, self.nt, nbatch)
                else:
                    raise NotImplementedError("Trajectory type not supported")
                advantages = self.P.get_advantages(returns, paths['tra'])

                # Policy updates
                for C in range(self.config.update_freq):
                    if self.config.use_baseline:
                        self.P.baseline.update_baseline(returns, paths['tra'])
                    self.P.update_policy(paths['obs'], paths['act'], advantages, old_logprobs=paths.get('old'))

                # Log the returns
                self.final_returns.append(returns[:, -1])
                self.save(epoch + 1)   # intermediately save the market maker for later loading

                # Plot if required
                if do_plot:
                    plot_WIM(paths, self.dt, title=f'epoch {epoch}', savename=self.config.out+'.png')

        self.logger.info("Training complete!")  # DONZO BONZO
        self.save(epoch + 1, True)
        self.plot()

class MarketMaker(UniformMarketMaker):
    def __init__(self, config: Config, inventory=0, wealth=0) -> None:
        super().__init__(config, inventory, wealth)
        self.book_quit = config.book_quit  # quit if book is empty
        self.config.logger.info(f"book_quit: {self.book_quit}")
        if config.book_quit:
            self.train = self.sparse_train
            self.get_paths = self.get_sparse_paths
            self.config.logger.info('using sparse training!')
    
    def get_sparse_paths(self, pbar, nt=0, nb=0, track_all=False) -> list[dict[list]]:
        """ get trajectories and compute rewards, only observing immediate state
        Inputs:
            - self.config.nbatch = number of trajectories to sample
            - track_all = also return (wealth, inventory, midprices) for WIM plot
            - plot_book = also plot (animate) book trajectory
        Output: 
            - paths: list of (nb) dictionaries each of list of (up to nt) steps
                (tra, obs, act, rew)
                if track_all: (tra, obs, act, rew, wea, inv, mid) 
                if PPO, also outputs log probs of actions in ['old'] """
        dt = self.dt
        if not nt: 
            nt = self.nt
            T = self.max_t
        else:
            T = nt*dt
        nbatch = nb if nb else self.config.nb

        paths = []
        avg_timestep = 0
        for b in range(nbatch):
            self.market.reset()
            W = self.market.W; I = self.market.I; midprice = 0
            trajectories = []
            observations = []
            actions = []
            rewards = []
            if self.do_ppo: logprobs = []
            if track_all:
                wealth = []
                inventory = []
                states = []
            # timestep
            terminated = False
            for t in range(nt):
                time_left = (T - t*dt,)
                state = self.market.state() + time_left
                if self.do_ppo:  # need to get log probability directly
                    action, logprob = self.P.policy.act(state, return_log_prob=True)
                    self.market.submit(*action)
                else:
                    action = self.market.act(state, self.P.policy.act)
                if self.book_quit and self.market.is_empty():
                    #self.logger.info(f'Batch {b} step {t}: Book is empty, quitting trajectory')
                    terminated = True
                    break   # quit if either bids or asks are empty
                dW, dI, midprice = self.market.step()
                if dW is ValueError:
                    lambda_sell = dI; lambda_buy = midprice
                    print(f"Batch {b} step {t} INVALID lambdas, ({lambda_sell}, {lambda_buy}), on book {self.market.book}")
                    self.market.reset(plot=True, make_bell=True)
                    break
                if self.book_quit and self.market.is_empty():
                    #self.logger.info(f'Batch {b} step {t}: Book is empty, quitting trajectory')
                    terminated = True
                    break   # quit if either bids or asks are empty
                observations.append(state)
                actions.append(action)
                if self.do_ppo: logprobs.append(logprob)
                reward_state = (dW, dI) + time_left
                rewards.append(self.market.reward(reward_state))
                trajectories.append(state + (dW, dI))
                W += dW; I += dI
                if track_all:
                    wealth.append(W)
                    inventory.append(I)
                    states.append((midprice, midprice-self.market.book.delta_b, midprice+self.market.book.delta_a))
            if not terminated or self.config.always_final:
                if len(rewards):
                    rewards[-1] += self.market.final_reward(W, I, self.market.book.midprice)
                else:
                    rewards = [self.market.final_reward(W, I, self.market.book.midprice)]
            pbar.update(1)
            path = {"tra": trajectories, "obs": observations, "act": actions, "rew": rewards, "val": W+I*midprice}
            if self.do_ppo:
                path["old"] = logprobs
            if track_all:
                path["wea"] = wealth
                path["inv"] = inventory
                path["book"] = states
            paths.append(path)
            avg_timestep += t+1
        avg_timestep /= nbatch
        self.logger.info(f"Average trajectory length: {avg_timestep}")
        return paths

    def sparse_train(self, plot_after=False):
        """ Train number of epochs x nbatch things
        - plot_after some # of epochs to show improvement? """
        nbatch = self.config.nb
        nepoch = self.config.ne - self.config.starting_epoch
        with tqdm(total=nepoch*nbatch) as pbar:  # make local pbar instance
            for epoch in range(self.config.starting_epoch, self.config.ne):
                do_plot = plot_after and ((epoch + 1) % plot_after == 0)
                pbar.set_description(f"Epoch {epoch+1}")
                paths = self.get_paths(pbar, nb=nbatch, track_all=do_plot)
                trajectories = np.concatenate([path['tra'] for path in paths], axis=0)
                observations = np.concatenate([path['obs'] for path in paths])
                actions = np.concatenate([path['act'] for path in paths])
                values = [path['val'] for path in paths]
                if self.do_ppo: logprobs = np.concatenate([path['old'] for path in paths])
                if self.config.trajectory == 'MC':
                    returns, finals = self.P.get_uneven_returns(paths)
                elif self.config.trajectory == 'TD':
                    returns, finals = self.P.get_uneven_td_returns(paths)
                else:
                    raise NotImplementedError("Trajectory type not supported")
                advantages = self.P.get_advantages(returns, trajectories)
                # first update will have old_logprobs = logprobs, so do 
                # C steps of policy updates on the same trajectories
                for C in range(self.config.update_freq):
                    if self.config.use_baseline:
                        self.P.baseline.update_baseline(returns, trajectories)
                    if self.do_ppo:
                        self.P.update_policy(observations, actions, advantages, logprobs)
                    else:
                        self.P.update_policy(observations, actions, advantages)
                # log the returns
                self.final_returns.append(finals)
                self.final_values.append(values)
                self.save(epoch+1, do_plot)   # intermediately save the market maker for later loading

                if do_plot:
                    plot_WIM(paths, self.dt, title=f'epoch {epoch}', savename=self.config.out+'.png')
        self.logger.info("DONZO BONZO!")
        self.save(epoch+1, True)
        self.plot()


if __name__ == "__main__":
    config = Config()
    config.set_name(0, make_new=True)
    print(f'loaded config {config.name}')
    MM = MarketMaker(config)
    #print(MM.P.policy.state_dict())
    MM.P.init_policy(MM.market,ne=1000,nb=100,new_train=False)
    out = config.out
    if os.path.exists(config.out+"_init-pol.pth"):
        os.remove(config.out+"_init-pol.pth")
    torch.save(MM.P.policy.state_dict(), config.out+"_init-pol.pth")
    print(f'saved policy to {config.out}_init-pol.pth')
    while 1:
        t = input('plot book path by pressing enter')
        if t: break
        MM.plot_book_path(nt=500, wait_time=0.2)