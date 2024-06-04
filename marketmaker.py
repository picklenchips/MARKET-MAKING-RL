import numpy as np
from config import Config
import torch
from base_market import Market
from policy import PPO
from util import np, get_logger, plot_WIM, export_plot, np2torch
from tqdm import tqdm
import os

class MarketMaker:
    def __init__(self, config: Config, inventory=0, wealth=0) -> None:
        self.config = config
        self.market = Market(inventory, wealth, config)
        self.logger = get_logger(config.log_out)
        self.P = PPO(config)
        self.PPO = True
        self.config.use_baseline = True
        self.dt = config.dt; self.max_t = config.max_t
        self.nt = config.nt
        self.final_returns = []  # track returns during training
    
    def get_paths(self, pbar, nt=None, nb=None, track_all=False):
        """ get trajectories and compute rewards, only observing immediate state
        Inputs:
        - self.config.nbatch = number of trajectories to sample
        - track_all = also return (wealth, inventory, midprices)
        Output:  dictionary of all np.ndarrays (tra, obs, act, rew)
        if track_all, output: (tra, obs, act, rew, wea, inv, mid) 
        if PPO, also outputs log probs of actions in ['old'] """
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
        if self.PPO:
            logprobs = np.empty((nbatch, nt))
        if track_all:  # track all for later plotting?
            wealth = np.empty((nbatch, nt))
            inventory = np.empty((nbatch, nt),dtype=int)
            midprices = np.empty((nbatch, nt))
        for b in range(nbatch):
            self.market.reset()
            W = self.market.W; I = self.market.I
            # timestep
            for t in range(nt):
                time_left = (T - t*dt,)
                state = self.market.state() + time_left
                if self.PPO:  # need to get log probability directly
                    action, logprob = self.P.policy.act(np.array(state), return_log_prob=True)
                    logprobs[b, t] = logprob
                    self.market.submit(*action)
                else:
                    action = self.market.act(state, self.P.policy.act)
                actions[b, t] = action
                dW, dI, midprice = self.market.step()
                reward_state = (dW, dI) + time_left
                rewards[b, t] = self.market.reward(reward_state)
                trajectories[b, t] = state + (dW, dI)
                if track_all:
                    W += dW; I += dI
                    wealth[b, t] = W
                    inventory[b, t] = I
                    midprices[b, t] = midprice
            rewards[b, t] += self.market.final_reward(dW, I, self.market.book.midprice)
            pbar.update(1)
        observations = trajectories[...,:obs_dim]
        paths = {"tra": trajectories, "obs": observations, "act": actions, "rew": rewards}
        if self.PPO:
            paths["old"] = logprobs
        if track_all:
            paths["wea"] = wealth
            paths["inv"] = inventory
            paths["mid"] = midprices
        return paths
    
    def get_td_lambda_returns(self, rewards, values, nt, nbatch):
        """ Compute TD(λ) returns """
        td_lambda_returns = np.zeros_like(rewards)
        for b in range(nbatch):
            G = rewards[b, -1] + self.config.discount * values[b, -1]
            for t in reversed(range(nt-1)):
                G = rewards[b, t] + self.config.discount * ((1 - self.config.lambd) * values[b, t+1] + self.config.lambd * G)
                td_lambda_returns[b, t] = G
        return td_lambda_returns


    def train(self, plot_after=False):
        """ Train number of epochs x nbatch things
        - plot_after some # of epochs to show improvement? """
        nbatch = self.config.nb
        nepoch = self.config.ne - self.config.starting_epoch
        with tqdm(total=nepoch*nbatch) as pbar:  # make local pbar instance
            for epoch in range(self.config.starting_epoch, self.config.ne):
                pbar.set_description(f"Epoch {epoch}", refresh=True)
                
                # Get paths and returns based on trajectory type
                if self.config.trajectory == 'MC':
                    paths = self.get_paths(pbar, nb=nbatch, track_all=plot_after)
                    returns = self.P.get_returns(paths['rew'])
                elif self.config.trajectory == 'TD':
                    paths = self.get_paths(pbar, nb=nbatch, track_all=plot_after)
                    values = self.P.baseline.network(np2torch(paths['tra'])).detach().cpu().numpy()
                    returns = self.get_td_lambda_returns(paths['rew'], values, self.nt, nbatch)
                else:
                    raise NotImplementedError("Trajectory type not supported")

                advantages = self.P.get_advantages(returns, paths['tra'])

                # Policy updates
                for C in range(self.config.update_freq):
                    self.P.baseline.update_baseline(returns, paths['tra'])
                    self.P.update_policy(paths['obs'], paths['act'], advantages, old_logprobs=paths.get('old'))

                # Log the returns
                self.final_returns.append(returns[:, -1])
                self.save(epoch + 1)   # intermediately save the market maker for later loading

                # Plot if required
                if plot_after and (epoch + 1) % plot_after == 0:
                    plot_WIM(paths['wea'], paths['inv'], paths['mid'], title=f'Epoch {epoch}', savename=self.config.out + '.png')

        self.logger.info("Training complete!")  # DONZO BONZO
        self.save(epoch + 1)
        self.plot()

# --- SAVE / LOAD --- #
    def plot(self):
        """ plot final scores and final path """
        finals = np.array(self.final_returns)
        print(finals.shape)
        export_plot(finals,"Scores",self.config.name,self.config.scores_plot)
        with tqdm(total=100) as pbar:
            path = self.get_paths(pbar, nb=100, track_all=True)
            plot_WIM(path['wea'], path['inv'], path['mid'], self.config.dt, title=f'Final Path (100 batches) {self.config.name}', savename=self.config.wim_plot)

    def save(self, epoch):
        """ Save a trained market maker model """
        old_out = self.config.out
        name, out = self.config.set_name(epoch)
        save_msg = f"[EPOCH {epoch}] Saved "
        if self.config.use_baseline:
            if os.path.exists(old_out+"_val.pth"): 
                os.remove(old_out+"_val.pth")
            torch.save(self.P.baseline.network.state_dict(), out+"_val.pth")
            save_msg += "baseline network, "
        if os.path.exists(old_out+"_pol.pth"): 
            os.remove(old_out+"_pol.pth")
        torch.save(self.P.policy.state_dict(), out+"_pol.pth")
        save_msg += "policy network, "
        if os.path.exists(old_out+"_scores.npy"): 
            os.remove(old_out+"_scores.npy")
        np.save(self.config.scores_out, np.array(self.final_returns))
        save_msg += f"final returns to {name} :)"
        self.logger.info(save_msg)

    def load(self):
        """ Return a trained market maker model from same config """
        name = self.config.out
        if not os.path.exists(name+"_pol.pth"):
            raise FileNotFoundError(f"Model {name} not found")
        if self.config.use_baseline:
            self.P.baseline.network.load_state_dict(torch.load(name+"_val.pth"))
            print(f"Loaded baseline network from {name}_val.pth")
        self.P.policy.load_state_dict(torch.load(name+"_pol.pth"))
        print(f"Loaded policy network from {name}_pol.pth")
        print(self.config.scores_out)
        self.final_returns = np.load(self.config.scores_out)