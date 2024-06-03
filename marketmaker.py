import numpy as np
from config import Config
import torch
from base_market import Market
from policy import PPO
from util import np, get_logger, plot_WIM, export_plot
from tqdm import tqdm
import os

class MarketMaker:
    def __init__(self, config: Config, inventory=0, wealth=0) -> None:
        self.config = config
        self.market = Market(inventory, wealth, config)
        self.logger = get_logger(config.log_path)
        self.P = PPO(config)
        self.PPO = True
        self.config.use_baseline = True
        self.dt = config.dt; self.max_t = config.max_t
        self.nt = config.nt
    
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
    
    #TODO: implement TD lambda shits? gonna need a different config too IG

    def train(self, plot_after=False):
        """ Train number of epochs x nbatch things
        - plot_after some # of epochs to show improvement? """
        final_rewards = []
        nepoch = self.config.ne; nbatch = self.config.nb
        with tqdm(total=nepoch*nbatch) as pbar:  # make local pbar instance
            for epoch in range(nepoch):
                pbar.set_description(f"On Epoch {epoch}",refresh=True)
                #TODO: TD Lambda insertion here
                if self.config.trajectory == 'MC':
                    paths = self.get_paths(pbar, nbatch, track_all=plot_after)
                else:
                    raise NotImplementedError
                pbar.set_description(f"Updating {epoch}",refresh=True)
                #pbar.set_postfix_str(f"Reward {rewards[b, -1]}", refresh=True)
                if plot_after:
                    if epoch + 1 % plot_after == 0:
                        plot_WIM(paths['wea'], paths['inv'], paths['mid'], title=f'epoch {epoch}')
                returns = self.P.get_returns(paths['rew'])
                advantages = self.P.get_advantages(returns, paths['tra'])
                # first update will have old_logprobs = logprobs, so do 
                # C steps of policy updates on the same trajectories
                for C in range(self.config.update_freq):
                    self.P.baseline.update_baseline(returns, paths['tra'])
                    self.P.update_policy(paths['obs'], paths['act'], advantages, old_logprobs=paths.get('old'))
                # log everything=
                rewards = paths['rew'][:,-1]
                avg_reward = np.mean(rewards)
                sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
                msg = "[EPOCH {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                        epoch, avg_reward, sigma_reward
                )
                final_rewards.append(rewards)
                self.logger.info(msg)
                self.save(epoch+1)   # intermediately save the market maker for later loading
                np.save(self.config.scores_output, final_rewards)
        final_rewards = np.array(final_rewards)
        self.logger.info("DONZO BONZO!")
        np.save(self.config.scores_output, final_rewards)
        export_plot(final_rewards,"Scores",self.config.name,self.config.out+'_scores.png')
        self.save(epoch+1)
        # plot last lil sample
        with tqdm(total=100) as pbar:
            path = self.get_paths(pbar, nb=100, track_all=True)
            plot_WIM(path['wea'], path['inv'], path['mid'], self.config.dt, title=f'Final Path (100 batches) {self.config.name}', savename=self.config.out+'_final.png')

# --- SAVE / LOAD --- #
    def save(self, epoch):
        """ Save a trained market maker model """
        old_out = self.config.out
        name, out = self.config.set_name(epoch)
        if self.config.use_baseline:
            if os.path.exists(old_out+"_val.pth"): os.remove(old_out+"_val.pth")
            torch.save(self.P.baseline.network.state_dict(), out+"_val.pth")
            self.logger.info(f"Saved baseline network to {name}_val.pth")
        if os.path.exists(old_out+"_pol.pth"): os.remove(old_out+"_pol.pth")
        torch.save(self.P.policy.state_dict(), out+"_pol.pth")
        self.logger.info(f"Saved policy network to {name}_pol.pth")

    def load(self):
        """ Return a trained market maker model from same config """
        name = self.config.out
        if self.config.use_baseline:
            self.P.baseline.network.load_state_dict(torch.load(name+"_val.pth"))
            print(f"Loaded baseline network from {name}_val.pth")
        self.P.policy.load_state_dict(torch.load(name+"_pol.pth"))
        print(f"Loaded policy network from {name}_pol.pth")