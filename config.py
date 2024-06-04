""" Object-oriented configuration file """
import os, math, pickle

def to_TF(value): return "T" if value else "F"
SAVEDIR = os.getcwd()+"/results"
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

class Config:
    def __init__(self, obs_dim=5, act_dim=4, n_layers=2, layer_size=10, 
                 lr=1e-3, discount=0.99, 
                 discrete=False, use_baseline=True, normalize_advantages=True, 
                 do_ppo = True, eps_clip=0.2, do_clip = True, entropy_coef = 0.02, 
                 nbatch=100, nepoch=1000, nt=10000, dt=1e-3, max_t=0, 
                 gamma=1, sigma=1e-2, trajectory='MC',
                 update_freq=5) -> None:
        # network stuff
        if trajectory == 'MC':
            # set default values for MC
            pass
        elif trajectory == 'TD':
            #TODO 
            self.lambd = 0.9 #? ? ? ?
            # set default values for TD?
            raise NotImplementedError
        self.book_quit = True   # END EARLY IF BOOK IS INVALID

        self.obs_dim = obs_dim  # policy input
        self.act_dim = act_dim  # policy output
        self.rew_dim = 2        # reward additional info
        self.val_dim = self.rew_dim + self.obs_dim  # baseline network input
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.discrete = discrete
        self.lr = lr
        self.trajectory = trajectory
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
        # PPO stuff
        self.do_ppo = do_ppo
        self.eps_clip = eps_clip          # clip between 1-eps_clip and 1+eps_clip
        self.do_clip = do_clip            # whether to clip the ratio
        self.entropy_coef = entropy_coef  # PPO entropy coefficient
        self.update_freq = update_freq    # how many times to gradient step in a row
        # savenames
    
    def set_name(self, epoch=None, make_new=False):
        """ update the configuration name """
        name = f"{self.trajectory}_{self.ne}-{self.nb}-{self.nt}_{to_TF(self.discrete)}{to_TF(self.use_baseline)}{to_TF(self.normalize_advantages)}{to_TF(self.do_clip)}"
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
    
    def save(self, filename=None):
        """ save a configuration to a filename"""
        if not filename:
            filename = self.out + "_config.pkl"
        filehandler = open(filename, 'w')
        pickle.dump(object, filehandler)

    def load(self, filename=None):
        """ load a configuration from a filename """
        if not filename:
            filename = self.out + "_config.pkl"
        filehandler = open(filename, 'r')
        return pickle.load(filehandler)

#TODO: just store the .pkl of the configs instead of re-creating from the filename
# bc with more complex configs, the filename will be a mess
# so use config.load() yeah?
def get_config(pathname: str) -> Config:
    """ Load a configuration from a filename """
    if "/" in pathname:
        pathname = pathname.split("/")[-1]
    config = Config()
    parts = pathname.split("_")
    if parts[0].isdigit():
        config.starting_epoch = int(parts[0])
        parts = parts[1:]
    config.trajectory = parts[0]
    config.ne, config.nb, config.nt = map(int, parts[1].split("-"))
    config.discrete = parts[2][0] == "T"
    config.use_baseline = parts[2][1] == "T"
    config.normalize_advantages = parts[2][2] == "T"
    config.do_clip = parts[2][3] == "T"
    config.set_name(config.starting_epoch)
    return config