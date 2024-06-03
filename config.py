""" Object-oriented configuration file """
import os, math

def to_TF(value): return "T" if value else "F"
SAVEDIR = os.getcwd()+"/results"
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

class Config:
    def __init__(self, obs_dim=5, act_dim=4, n_layers=2, layer_size=10, 
                 lr=1e-3, discount=0.99, 
                 discrete=False, use_baseline=True, normalize_advantages=True, 
                 eps_clip=0.2, do_clip = True, entropy_coef = 0.02, 
                 nbatch=10, nepoch=100, nt=1000, dt=1e-3, max_t=0, 
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
        self.eps_clip = eps_clip          # clip between 1-eps_clip and 1+eps_clip
        self.do_clip = do_clip            # whether to clip the ratio
        self.entropy_coef = entropy_coef  # PPO entropy coefficient
        self.update_freq = update_freq    # how many times to gradient step in a row
        # savenames
        log_name = f"{trajectory}_{self.ne}-{self.nb}-{self.nt}_{to_TF(discrete)}{to_TF(use_baseline)}{to_TF(normalize_advantages)}"
        # make new directory to store results
        L = len(log_name)
        i = 0; dontmakedir=False
        while os.path.exists(SAVEDIR+"/"+log_name):
            with open(SAVEDIR+"/"+log_name+"/"+log_name+".log", "r") as f:
                if f.readline() == '':
                    dontmakedir=True
                    break
            log_name = log_name[:L]+str(i)
            i += 1
        if not dontmakedir: os.mkdir(SAVEDIR+"/"+log_name)
        self.save_dir = SAVEDIR+"/"+log_name+"/"
        self.name = log_name
        self.out = self.save_dir + log_name
        self.log_path = self.save_dir+log_name + ".log"
        self.scores_output = self.save_dir+log_name+"_scores.npy"
        self.plot_output = self.save_dir+log_name+"_rewards.png"

def get_config(pathname: str) -> Config:
    """ Load a configuration from a file """
    if "/" in pathname:
        pathname = pathname.split("/")[-1]
    config = Config()
    parts = pathname.split("_")
    if len(parts) >= 3:
        return False
    config.trajectory = parts[0]
    config.ne, config.nb, config.nt = map(int, parts[1].split("-"))
    config.discrete = parts[2][0] == "T"
    config.use_baseline = parts[2][1] == "T"
    config.normalize_advantages = parts[2][2] == "T"
    return config