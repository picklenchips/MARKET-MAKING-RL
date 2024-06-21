""" Object-oriented configuration file, used to define all hyperparameters 
    also defines how to save, load, resume past configs/runs """
import os, math, pickle
from glob import glob
import argparse, re
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
    def __init__(self, save_config=False,  # set_name and save after making?
                 # training parameters
                 nbatch=100, nepoch=1000, nt=10000, dt=1e-3, max_t=0, update_freq=5,
                 # TODO: generalize an actual TD lambda algorithm for this
                 past_obs=2, 
                 # network parameters
                 obs_dim=5, act_dim=4, rew_dim=2, n_layers=2, layer_size=10, 
                 n_obs = 1,  # number of past observations to use for policy
                 # learning parameters
                 lr=0.001, discount=0.9999, lambd=0.9, 
                 # reward / return parameters
                 immediate_reward=True, add_time=False, add_inventory=False, always_final=False,
                 use_baseline=True, normalize_advantages=True, trajectory='MC', book_quit = True,
                 # policy parameters
                 discrete=False, do_ppo = True, do_clip = True, eps_clip=0.2, entropy_coef = 0.00, 
                 # initial LOB parameters
                 midprice = 533, spread = 10, nstocks = 10000, make_bell = True,
                 nsteps = 1000, substeps = 1,   # if not make_bell, also perform some random stock things
                 # market dynamics parameters - https://www.desmos.com/calculator/r06ektyu4w 
                 #         a     d     f     b     c      g
                 betas = (7.2, -2.13, -0.8, -2.3, 0.167, -0.1), max_dW = 3000,
                 # OLD AVELLANEDA PARAMETERS
                 gamma=1, sigma=1e-2, 
                 plot_intermediate=True) -> None:
        # network stuff
        
        #TODO: generalize TD lambda to past obs?
        self.past_obs = past_obs  # number of past observations to include
        ######### NETWORK PARAMETERS ##########
        self.n_obs = n_obs  # number of past observations to use for policy
        # policy input is 4 * n_obs + 1, as we add timeleft after observing n_obs observations
        self.obs_dim = obs_dim*n_obs + 1  # policy input
        self.act_dim = act_dim  # policy output
        self.rew_dim = rew_dim   # reward additional info
        self.val_dim = self.rew_dim + self.obs_dim  # baseline network input
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.network_out = INITSAVE+"/"+str(self.n_obs)+"*"+"-".join(map(str, [self.obs_dim, self.act_dim, self.n_layers, self.layer_size]))+"_init-pol.pth"
        ######### RETURN PARAMETERS ##########
        self.lr = lr
        self.discount = discount     # discount for computing returns
        self.lambd = lambd           # for TD lambda
        self.trajectory = trajectory # MC or TD
        self.discrete = discrete     # discrete or continuous action space
        self.book_quit = book_quit   # END EARLY IF BOOK IS INVALID
        self.update_freq = update_freq    # how many times to gradient step in a row
        if self.discrete:
            raise NotImplementedError("Categorical policy not implemented yet")
        ######### TRAINING PARAMETERS ##########
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
        ##### REWARD STUFF ####
        self.add_inventory = add_inventory  # add inventory to immediate reward
        self.add_time = add_time  # subtract time from immediate reward
        self.immediate_reward = immediate_reward  # use immediate reward function
        self.always_final = always_final  # always add final reward, regardless of at max_t or not
        ##### POLICY STUFF ####
        self.use_baseline = use_baseline  # use baseline network to compute actual advantages
        self.normalize_advantages = normalize_advantages  # normalize advantages
        self.do_ppo = do_ppo              # use full PPO?
        self.eps_clip = eps_clip          # clip between 1-eps_clip and 1+eps_clip
        self.do_clip = do_clip            # whether to clip the ratio
        self.entropy_coef = entropy_coef  # PPO entropy coefficient
        # plot intermediate scores, values always?
        self.plot_intermediate = plot_intermediate  # plot intermediate results after every epoch
        #### MARKET LOB STUFF #####
        self.midprice = midprice   # DEFINE HOW LOB STARTS
        self.spread = spread
        self.nstocks = nstocks
        self.make_bell = make_bell  # make a bell curve for each side of LOB to start
        self.nsteps = nsteps
        self.substeps = substeps
        self.betas = betas         # DEFINING MARKET STEPS
        self.max_dW = max_dW
        self.sigma = sigma  # avellaneda param
        self.gamma = gamma  # avellaneda param
        
        self.run_no = 0  # used to run repeat stuffs
        if save_config:
            self.set_name(make_new=True)
            self.save()
    
    def __str__(self):
        """ return string of all current parameters """
        charsPerLine = 70; rowstart = ''
        thingsPerLine = 5
        ret = "Config: \n"
        imported_stuff = dir(self)
        #type_dict = defaultdict(list)  # list of tuples for each type
        stuff = []
        for name in imported_stuff:
            if not name.startswith('__'):  # ignore the default namespace variables
                typ = str(type(eval("self."+name))).split("'")[1]
                if typ == 'method':   # ignore functions
                    continue
                stuff.append((typ,name,eval("self."+name)))
        #stuff.sort(key=lambda x: x[1])  # sort alphabetically by name
        stuff.sort(key =lambda x: x[0])  # sort by type
        row = rowstart
        i = 0
        for id in stuff:
            thing = id[2]
            if isinstance(thing, str):  # cut off any directories from names
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
            name = policypath.split("/")[-1]
            epoch = int(name.split("_")[0])
            config.starting_epoch = epoch
            run_no = name.split("_")[-2]
            if run_no.isdigit():
                config.run_no = int(run_no)
            else:
                config.run_no = 0            
        config.set_name(config.starting_epoch)
        return config
    
    def set_name(self, epoch=0, make_new=False, save_dir=SAVEDIR):
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
            try:
                if self.add_inventory:
                    rew_str += "-I"
                if self.add_time:
                    rew_str += "-T"
            except AttributeError:
                rew_str += "-I-T"
        bool_strs = [pol_str, rew_str]
        bools = [self.discrete, self.use_baseline, self.normalize_advantages, self.do_clip, self.book_quit, self.add_time, self.immediate_reward, self.always_final]
        # OLD NAME
        old_name = '_'.join(['-'.join(strs),'-'.join(map(str, ints)),''.join(map(to_TF, bools))])
        # NEW NAME
        name = '_'.join(['-'.join(strs),'-'.join(map(str, ints)),'_'.join(bool_strs)])
        # make new directory to store results
        L = len(name)
        # if model is fully complete, don't overwrite
        if make_new:
            i = 0
            while os.path.exists(f"{save_dir}/{name}/{self.ne}_{name}_pol.pth"):
                i += 1
                name = name[:L]+f"_{i}"
            self.run_no = i
        else:
            try:
                if self.run_no:
                    name += f"_{self.run_no}"
            except AttributeError:
                pass
        if not os.path.exists(f"{save_dir}/{name}"):
            os.mkdir(f"{save_dir}/{name}")
        # SET NAMES
        self.base_name = name
        self.save_dir = f"{save_dir}/{name}/"
        if epoch:   # number of epochs completed
            name = f"{epoch}_" + name
            old_name = f"{epoch}_" + old_name
        self.name = name
        self.base_out = self.save_dir + self.base_name
        self.out  = self.save_dir + name
        self.val_out = self.out+'_val.pth'
        self.best_val_out = self.out+'_best_val.pth'
        self.best_pol_out = self.out+'_best_pol.pth'
        self.pol_out = self.out+'_pol.pth'
        self.scores_out  = self.out+'_scores.npz'
        self.scores_plot = self.out+'_scores.png'
        self.values_out  = self.out+'_values.npz'
        self.values_plot = self.out+'_values.png'
        self.wim_plot    = self.out+'_wim.png'
        self.log_out     = self.base_out+".log"
        self.logger = get_logger(self.log_out)

        self.old_name = old_name
        self.old_out = self.save_dir + old_name
        self.old_val_out = self.old_out+'_val.pth'
        self.old_pol_out = self.old_out+'_pol.pth'
        self.old_scores_out  = self.old_out+'_scores.npz'
        self.old_scores_plot = self.old_out+'_scores.png'
        self.old_values_out  = self.old_out+'_values.npz'
        self.old_values_plot = self.old_out+'_values.png'
        self.old_wim_plot    = self.old_out+'_wim.png'
        self.old_log_out     = self.old_out+".log"
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
    is_old_name = True
    for ch in pol_str:
        if ch not in "FT":
            is_old_name = False
            break
    if is_old_name:
        print(pol_str)
        bools = [ch == "T" for ch in pol_str]
        config.discrete, config.use_baseline, config.normalize_advantages, config.do_clip, config.book_quit, config.add_time, config.immediate_reward = bools
        config.always_final = False
    else:
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
    if len(glob(filepath)) == 0:
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

def update_epoch(save_dir: str, epoch: int) -> str:
    """ rename directory + all of its (non .png) files 
    to reflect a new epoch
    Returns the new directory name """
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    prevdir = '/'.join(save_dir.split("/")[:-1])
    old_base = save_dir.split("/")[-1]
    new_base = re.sub(r"_(\d+)-", f"_{epoch}-", old_base)
    for f in glob(save_dir+"/*"):
        name = f.split("/")[-1]
        if f[-3:] == 'png': 
            continue
        if old_base not in f:
            continue
        newname = re.sub(r"_(\d+)-", f"_{epoch}-", name)
        os.rename(f, save_dir+"/"+newname)
    print(save_dir, prevdir+"/"+new_base)
    os.rename(save_dir, prevdir+"/"+new_base)
    return prevdir+"/"+new_base

def get_config(args: argparse.ArgumentParser) -> Config:
    """ Load a configuration from a filename """
    config = False
    if not args: return Config()
    if args.load:  # load a model from a filepath directory
        # FIRST, LOOK FOR PKL FILES
        config, found_pkl = search_for_config(args.load)
        if not config:
            raise FileNotFoundError(f"Could not find config file from {args.load}")
        if not found_pkl: config.save()
        if args.expand:
            config.ne += args.expand
            update_epoch(config.save_dir, config.ne)
            config.set_name(epoch=config.starting_epoch)
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
    if args.book_size: config.nstocks = args.book_size
    if args.update_freq: config.update_freq = args.update_freq
    if args.n_obs: config.n_obs = args.n_obs
    
    config.set_name()
    # see if there is an existing plot w the same thing
    # make_new = make new directory if directory exists
    oldconfig, found_pkl = search_for_config(config.save_dir)
    make_new = not (args.plot or args.expand)
    print(oldconfig.name)
    if oldconfig:
        # old directory is done and we want to make a new one
        if oldconfig.starting_epoch == config.ne and make_new:
            config.set_name(make_new=True)
            config.save()
            return config
        config = oldconfig
        config.set_name(config.starting_epoch)
        make_new = make_new and not (config.starting_epoch and config.starting_epoch < config.ne)
    policypath = glob(config.save_dir+"/*_pol.pth")
    if len(policypath):
        policypath = policypath[0]
        config.starting_epoch = int(policypath.split("/")[-1].split("_")[0])
    if config.starting_epoch == config.ne:
        if args.expand:
            config.ne += args.expand
            update_epoch(config.save_dir, config.ne)
            config.set_name(epoch=config.starting_epoch, make_new=make_new)
        else:
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
        config, found_pkl = search_for_config(t)
        print(config)