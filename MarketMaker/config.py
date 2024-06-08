""" Object-oriented configuration file """
import os, math, pickle
from collections import defaultdict
from glob import glob
import argparse
try:
    from MarketMaker.util import get_logger
except ModuleNotFoundError:
    from util import get_logger

to_TF = lambda value: "T" if value else "F"
full_TF = lambda value, label: "no-"+label if not value else label

SAVEDIR = os.getcwd()+"/results"
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)
INITSAVE = os.getcwd() +"/init_policies"
if not os.path.exists(INITSAVE):
    os.mkdir(INITSAVE)

class Config:
    def __init__(self, obs_dim=5, act_dim=4, rew_dim=2, 
                 n_layers=2, layer_size=10, lr=0.1, discount=0.9999, 
                 immediate_reward=True, add_time=False, always_final=False, add_inventory=False,
                 use_baseline=True, normalize_advantages=True,
                 discrete=False, do_ppo = True, book_quit = True,
                 eps_clip=0.2, do_clip = True, entropy_coef = 0.00, 
                 nbatch=100, nepoch=1000, nt=10000, dt=1e-3, max_t=0, 
                 gamma=1, sigma=1e-2, trajectory='MC', past_obs=2,
                 update_freq=5, lambd=0.9, save_config=False,
                 plot_intermediate=True) -> None:
        # network stuff
        self.book_quit = book_quit   # END EARLY IF BOOK IS INVALID
        #TODO: generalize TD lambda to past obs?
        self.past_obs = past_obs  # number of past observations to include
        self.obs_dim = obs_dim  # policy input
        self.act_dim = act_dim  # policy output
        self.rew_dim = rew_dim   # reward additional info
        self.val_dim = self.rew_dim + self.obs_dim  # baseline network input
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.network_out = INITSAVE+"/"+"-".join(map(str, [self.obs_dim, self.act_dim, self.n_layers, self.layer_size]))+"_init-pol.pth"

        self.lr = lr
        self.discrete = discrete
        self.trajectory = trajectory
        self.lambd = lambd   # for TD lambda
        # set time parameters. np.arange(0, max_t, dt) | np.linspace(0, max_t, nt)
        self.starting_epoch = 0
        self.ne = nepoch
        self.nb = nbatch
        self.dt = dt
        if max_t:  # time is specified
            self.nt = math.ceil(max_t/dt)
        else:     # nt is specified
            max_t = nt*dt
            self.nt = nt
        self.max_t = max_t
        # market stuff
        self.sigma = sigma  # avellaneda param 
        self.gamma = gamma  # avellaneda param
        # reward stuff
        self.add_inventory = add_inventory  # add inventory to immediate reward
        self.use_baseline = use_baseline  # use baseline network to compute actual advantages
        self.normalize_advantages = normalize_advantages  # normalize advantages
        self.discount = discount          # discount for computing returns
        self.add_time = add_time  # subtract time from immediate reward
        self.immediate_reward = immediate_reward  # use immediate reward function
        # PPO stuff
        self.do_ppo = do_ppo
        self.eps_clip = eps_clip          # clip between 1-eps_clip and 1+eps_clip
        self.do_clip = do_clip            # whether to clip the ratio
        self.entropy_coef = entropy_coef  # PPO entropy coefficient
        self.update_freq = update_freq    # how many times to gradient step in a row
        self.always_final = always_final  # always add final reward

        self.plot_intermediate = plot_intermediate  # plot intermediate results after every epoch
        if save_config:
            self.set_name(make_new=True)
            self.save()
    
    def __str__(self):
        """ return string version of config """
        charsPerLine = 70; rowstart = ''
        thingsPerLine = 5
        ret = "Config: \n"
        imported_stuff = dir(self)
        type_dict = defaultdict(list)  # list of tuples for each type
        stuff = []
        for name in imported_stuff:
            if not name.startswith('__'):  # ignore the default namespace variables
                typ = str(type(eval("self."+name))).split("'")[1]
                if typ == 'method':   # ignore functions
                    continue
                stuff.append((typ,name,eval("self."+name)))
        #stuff.sort(key=lambda x: x[1])  # sort alphatbetically by name
        stuff.sort(key =lambda x: x[0])  # sort by type
        row = rowstart
        i = 0
        for id in stuff:
            thing = id[2]
            if isinstance(thing, str):
                if thing[-1] == '/':
                    thing = thing[:-1]
                thing = thing.split("/")[-1]
            new = f"{id[1]} = {thing}, "
            if len(row) + len(new) - 2 > charsPerLine or i == thingsPerLine:
                ret += row[:-2] + '\n'
                row = rowstart
                i = 0
            row += new
            i += 1
        return ret

    def save(self, filename=None):
        """ save a configuration to a filename"""
        if not filename:
            filename = self.base_out + "_config.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename=None):
        """ load a configuration from a filename """
        if not filename:
            filename = self.out + "_config.pkl"
        config = False
        with open(filename, 'rb') as f:
            config = pickle.load(f)
        dirabove = '/'.join(filename.split('/')[:-1])
        policypath = glob(dirabove+"/*_pol.pth")
        if len(policypath):
            policypath = policypath[0]
            config.starting_epoch = int(policypath.split("/")[-1].split("_")[0])
        config.set_name(config.starting_epoch)
        return config
    
    def set_name(self, epoch=0, make_new=False):
        """ update the configuration name. needed for naming after each epoch for correct 
        saving and loading of value, policy networks. """
        self.network_out = INITSAVE+"/"+"-".join(map(str, [self.obs_dim, self.act_dim, self.n_layers, self.layer_size]))+"_init-pol.pth"
        strs = [self.trajectory]
        ints = [self.ne, self.nb, self.nt]
        floats = []
        pol_str = "gaus-" if not self.discrete else "categ-"
        pol_str += "reinforce" if not self.do_ppo else ("ppo-clip" if self.do_clip else "ppo")
        pol_str += "-rets" if not self.use_baseline else ("-norm-adv" if self.normalize_advantages else "-adv")
        rew_str = "liquid" if self.always_final else ""
        if self.immediate_reward:
            if self.always_final: rew_str += "-"
            rew_str += "W"
            if self.add_inventory:
                rew_str += "-I"
            if self.add_time:
                rew_str += "-T"
        bool_strs = [pol_str, rew_str]
        bools = [self.discrete, self.use_baseline, self.normalize_advantages, self.do_clip, self.book_quit, self.add_time, self.immediate_reward, self.always_final]
        # OLD NAME
        name = '_'.join(['-'.join(strs),'-'.join(map(str, ints)),''.join(map(to_TF, bools))])
        # NEW NAME
        name = '_'.join(['-'.join(strs),'-'.join(map(str, ints)),'_'.join(bool_strs)])
        # make new directory to store results
        L = len(name)
        i = 0  # duplicate models?
        # if model is fully complete, don't overwrite
        if make_new:
            while os.path.exists(f"{SAVEDIR}/{name}/{self.ne}_{name}_pol.pth"):
                name = name[:L]+str(i)
                i += 1
        if not os.path.exists(f"{SAVEDIR}/{name}"):
            os.mkdir(f"{SAVEDIR}/{name}")
        # SET NAMES
        self.base_name = name
        self.save_dir = f"{SAVEDIR}/{name}/"
        if epoch:   # number of epochs completed
            name = f"{epoch}_" + name
        self.name = name
        self.base_out = self.save_dir + self.base_name
        self.out  = self.save_dir + name
        self.val_out = self.out+'_val.pth'
        self.pol_out = self.out+'_pol.pth'
        self.scores_out  = self.out+'_scores.npz'
        self.scores_plot = self.out+'_scores.png'
        self.values_out  = self.out+'_values.npz'
        self.values_plot = self.out+'_values.png'
        self.wim_plot    = self.out+'_wim.png'
        self.log_out     = self.base_out+".log"
        self.logger = get_logger(self.log_out)
        return self.name, self.out

#TODO: just store the .pkl of the configs instead of re-creating from the filename
# bc with more complex configs, the filename will be a mess
# so use config.load() yeah?
def config_from_name(pathname: str) -> Config | FileNotFoundError:
    """ return config from its name by reversing set_name """
    if "/" in pathname:
        if pathname[-1] == '/':
            pathname = pathname[:-1]
    parts = pathname.split("/")[-1].split("_")
    if len(parts) < 3: raise FileNotFoundError(f"{pathname} is not a config path")
    config = Config()
    if parts[0].isdigit():
        config.starting_epoch = int(parts[0])
        parts = parts[1:]
    config.trajectory = parts[0]
    config.ne, config.nb, config.nt = map(int, parts[1].split("-"))
    pol_str = parts[2]
    config.discrete = not "gaus" in pol_str
    config.do_ppo = "ppo" in pol_str
    config.do_clip = "clip" in pol_str
    config.normalize_advantages = "norm" in pol_str
    config.use_baseline = "adv" in pol_str
    if len(parts) > 3:
        rew_str = parts[3]
        config.always_final = "liquid" in rew_str
        config.immediate_reward = "W" in rew_str
        config.add_inventory = "I" in rew_str
        config.add_time = "T" in rew_str
    else:
        config.always_final = config.immediate_reward = False
    # load from string
    config.set_name(config.starting_epoch)
    return config

def search_for_config(filepath: str) -> Config | bool:
    """ Load a configuration from a filename
    finds pickle files, and then sets config from directory name """
    if not filepath.startswith(SAVEDIR):
        filepath = SAVEDIR + "/" + filepath
    for f in glob(filepath):
        # first, look for pickle files
        if f[-4] == ".pkl":
            return Config().load(f), True
        if os.path.isdir(f):
            if f[-1] != "/":
                f += "/"
            for f1 in glob(f+"*_config.pkl"):
                return Config().load(f1), True
            try: 
                return config_from_name(f), False
            except FileNotFoundError:
                pass
        else:
            dirabove = '/'.join(f.split('/')[:-1])
            try:
                return config_from_name(dirabove), False
            except FileNotFoundError:
                pass
    return False, False

def get_config(args: argparse.ArgumentParser) -> Config:
    """ Load a configuration from a filename """
    config = False
    if not args: return Config()
    if args.load:  # load a model from a filepath directory
        # FIRST, LOOK FOR PKL FILES
        config, found_pkl = search_for_config(args.load)
        if not config:
            raise FileNotFoundError(f"Could not find config file from {args.load}")
        if not found_pkl:
            config.save()
        return config
    # SET DEFAULT CONFIG
    config = Config(save_config=False, do_ppo=(not args.noppo), use_baseline = (not args.noadv), do_clip=(not args.noclip),
                    immediate_reward=(not args.no_immediate), book_quit=(not args.uniform), add_inventory=args.add_inventory,
                    add_time=args.add_time, always_final=args.always_final)
    if args.td: config.trajectory = 'TD'
    if args.nb: config.nb = args.nb
    if args.nt: config.nt = args.nt
    if args.ne: config.ne = args.ne
    
    config.set_name()
    # see if there is an existing plot w the same thing
    # make_new = make new directory if directory exists
    oldconfig, found_pkl = search_for_config(config.save_dir)
    make_new = not (args.resume or args.plot)
    print(oldconfig.name)
    if oldconfig:
        config = oldconfig
        config.set_name(config.starting_epoch)
        make_new = make_new and not (config.starting_epoch and config.starting_epoch < config.ne)
    policypath = glob(config.save_dir+"/*_pol.pth")
    if len(policypath):
        policypath = policypath[0]
        config.starting_epoch = int(policypath.split("/")[-1].split("_")[0])
    if config.starting_epoch == config.ne:
        config.set_name(0,make_new=make_new)
        config.save()
    else:
        config.set_name(config.starting_epoch)
    if not found_pkl:
        config.save()
    return config

# test config naming
if __name__ == "__main__":
    while 1:
        t = input("Type config name to load from: ")
        config = search_for_config(t)
        print(config)