""" Object-oriented configuration file """
import os, math, pickle
import logging
from collections import defaultdict
from glob import glob
import argparse

def to_TF(value): return "T" if value else "F"
SAVEDIR = os.getcwd()+"/results"
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

class Config:
    def __init__(self, obs_dim=5, act_dim=4, rew_dim=2, n_layers=2, layer_size=10, 
                 lr=1e-3, discount=0.99, subtract_time=False, immediate_reward=False,
                 discrete=False, use_baseline=True, normalize_advantages=True, 
                 do_ppo = True, eps_clip=0.2, do_clip = True, entropy_coef = 0.02, 
                 nbatch=100, nepoch=1000, nt=10000, dt=1e-3, max_t=0, 
                 gamma=1, sigma=1e-2, trajectory='MC',
                 update_freq=5, lambd=0.9, save_config=True) -> None:
        # network stuff
        self.book_quit = True   # END EARLY IF BOOK IS INVALID

        self.obs_dim = obs_dim  # policy input
        self.act_dim = act_dim  # policy output
        self.rew_dim = 2        # reward additional info
        self.val_dim = self.rew_dim + self.obs_dim  # baseline network input
        self.n_layers = n_layers
        self.layer_size = layer_size

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
        self.use_baseline = use_baseline  # use baseline network to compute actual advantages
        self.normalize_advantages = normalize_advantages  # normalize advantages
        self.discount = discount          # discount for computing returns
        self.subtract_time = subtract_time  # subtract time from immediate reward
        self.immediate_reward = immediate_reward  # use immediate reward function
        # PPO stuff
        self.do_ppo = do_ppo
        self.eps_clip = eps_clip          # clip between 1-eps_clip and 1+eps_clip
        self.do_clip = do_clip            # whether to clip the ratio
        self.entropy_coef = entropy_coef  # PPO entropy coefficient
        self.update_freq = update_freq    # how many times to gradient step in a row
        if save_config:
            self.set_name(make_new=True)
            self.save()
    
    def set_name(self, epoch=None, make_new=False):
        """ update the configuration name """
        strs = [self.trajectory]
        ints = [self.ne, self.nb, self.nt]
        floats = []
        bools = [self.discrete, self.use_baseline, self.normalize_advantages, self.do_clip, self.book_quit, self.subtract_time, self.immediate_reward]
        name = '_'.join(['-'.join(strs),'-'.join(map(str, ints)),''.join(map(to_TF, bools))])
        # make new directory to store results
        L = len(name)
        i = 0  # duplicate models?
        # if model is fully complete, don't overwrite
        if make_new:
            while os.path.exists(f"{SAVEDIR}/{name}/{self.ne}_{name}_val.pth"):
                name = name[:L]+str(i)
                i += 1
        if not os.path.exists(f"{SAVEDIR}/{name}"):
            os.mkdir(f"{SAVEDIR}/{name}")
        # SET NAMES
        self.save_dir = f"{SAVEDIR}/{name}/"
        self.base_name = name
        if isinstance(epoch, int):   # number of epochs completed
            name = f"{epoch}_" + name
        self.name = name
        self.out  = self.save_dir + name
        self.scores_out = self.out+'_scores.npy'
        self.scores_plot = self.out+'_scores.png'
        self.wim_plot   = self.out+'_wim.png'
        self.log_out    = self.save_dir+self.base_name+".log"
        return self.name, self.out
    
    def print(self):
        """ print the configuration to a string """
        charsPerLine = 70; rowstart = ''
        thingsPerLine = 1
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
            new = f"{id[1]} = {id[2]}, "
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
            filename = self.out + "_config.pkl"
        filehandler = open(filename, 'wb')
        pickle.dump(self, file=filehandler)
        #filehandler.close() 

    def load_config(self, filename=None):
        """ load a configuration from a filename """
        if not filename:
            filename = self.out + "_config.pkl"
        filehandler = open(filename, 'r')
        config = pickle.load(filehandler)
        #filehandler.close()
        return config

#TODO: just store the .pkl of the configs instead of re-creating from the filename
# bc with more complex configs, the filename will be a mess
# so use config.load() yeah?
def get_config(args: argparse.ArgumentParser) -> Config:
    """ Load a configuration from a filename """
    config = False; pathname = ''
    if args.load:  # load a model from a filepath directory
        # FIRST, LOOK FOR PKL FILES
        for f in glob(SAVEDIR+"/"+args.load):
            if f[-4] == ".pkl":
                config = pickle.load(f)
                break
            if os.path.isdir(f):
                if f[-1] != "/":
                    f += "/"
                for f1 in glob(f+"*_config.pkl"):
                    config = pickle.load(f1)
                    break
                # ELSE JUST ROLL WITH DIRECTORY NAME
        pathname = f
        if not config:
            config = Config(save_config=False)
    # SET DEFAULT CONFIG
    else:
        config = Config(save_config=False)
        # if we add other policies LOL
        if args.td: config.trajectory = 'TD'
        if args.nb: config.nb = args.nb
        if args.nt: config.nt = args.nt
        if args.ne: config.ne = args.ne
        if args.noppo: config.do_ppo = False
        if args.noadv: config.use_baseline = False
        if args.noclip: config.do_clip = False
        if args.immediate: config.immediate_reward = True
    
    config.set_name(make_new=(not args.resume and not args.plot and not args.load))
    if args.resume or args.plot:
        pathname = config.save_dir
    if not pathname:
        config.save()
        return config
    if "/" in pathname:
        pathname = pathname.split("/")[-1]
    parts = pathname.split("/")[-1].split("_")
    if parts[0].isdigit():
        if not config:
            config = Config()
            config.starting_epoch = int(parts[0])
            parts = parts[1:]
        else:  # config has been loaded previously
            config.starting_epoch = int(parts[0])
            config.set_name(config.starting_epoch)
            return config
    # GET CONFIG FROM PATHNAME ONLY
    config.trajectory = parts[0]
    config.ne, config.nb, config.nt = map(int, parts[1].split("-"))
    config.discrete = parts[2][0] == "T"
    config.use_baseline = parts[2][1] == "T"
    config.normalize_advantages = parts[2][2] == "T"
    config.do_clip = parts[2][3] == "T"
    if len(parts[2]) > 4:
        config.book_quit = parts[2][4] == "T"
    if len(parts[2]) > 5:
        config.subtract_time = parts[2][5] == "T"
    if len(parts[2]) > 6:
        config.immediate_reward = parts[2][6] == "T"
    # load from string
    config.set_name(config.starting_epoch)
    config.save()
    return config

# test config naming
if __name__ == "__main__":
    config = Config()
    print(config.print())