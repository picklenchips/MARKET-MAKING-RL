from MarketMaker.marketmaker import MarketMaker
from MarketMaker.config import get_config
from MarketMaker.util import np, torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load",   help='load a model from a results/name directory')
parser.add_argument("-r", "--resume", action='store_true', help='resume training from a model')
parser.add_argument("-p", "--plot", action='store_true', help='dont train, just plot')

parser.add_argument("--policy", help='use noppo to turn ppo off. use discrete to use discrete instead of \
                                      gaussian probability', default='ppo')
parser.add_argument("-ne", "--ne", help="number of epochs to train policy for", type=int)
parser.add_argument("-nb", "--nb", help="number of trajectories to sample for each policy update", type=int)
parser.add_argument("-nt", "--nt", help="number of timesteps to progress trajectory for", type=int)
parser.add_argument("-lr", "--lr", help="learning rate for the policy (Adam) optimizer", type=float)
parser.add_argument("-opt", "--optimizer", help="String name for the optimizer (like Adam, AdamW, SGD... case sensitive)", type=float)
parser.add_argument("--noclip", help='dont clip the ratio in PPO', action='store_true', 
                    )
parser.add_argument("--noppo",  help='dont use PPO, just use regular policy grad', action='store_true', 
                    )
parser.add_argument("--noadv",  help='dont use advantages, just returns', action='store_true', 
                    )
parser.add_argument( '--nobookquit','--uniform', help='dont quit early', action='store_true', 
                    )
parser.add_argument("--immediate", help='use immediate rewards', action='store_true',)
parser.add_argument("--subtract_time", "-st", help='subtract time from reward', action='store_true', )
parser.add_argument("--seed", default=0, type=int)
# TD LAMBDA STUFF?
parser.add_argument("--td", "--TD", "-td", action='store_true')

if __name__ == "__main__":
    """
    See help for more information
    TODO: put the parse information into the README.md
    """
    args = parser.parse_args()
    # 
    #   SET A SEED? Feel like we need complete randomness for this though... is there anyway to 
    #        dynamically change the seed as we progress through different epochs?       
    #
    set_seed = False
    if set_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    """ Initialize a configuration file, storing all the hyperparameters for the market
        - get_config(args) returns a Config object that correctly matches args.
            - will automatically resume an unfinished training session if args matches a config 
                in results/
            - args.load"""
    config = get_config(args)
    print(config)

    """ Initialize an instance of a MarketMaker as MM
    Functions:
    - MM.load(): load a market that matches the current config 
    - MM.save(): save the current policy, value networks. plot performance
    - MM.plot(): plot the trained policy on the market & performance
        args:
        - plot_book: iterate through the book over a single trajectory 
            - wait_time: time to wait between seeing each LOB instance
        - nt: number of timesteps in trajectory
        - nb: number of batches to plot for final trajectory (4)plot
    - MM.train(): train the policy on the market
        args:
        - plot_after: plot the market after this many epochs-
    - MM.get_paths():
        args:
        - 
    MM initializes with a Config, RewardMarket, and PPO or PolicyGradient objects.
    """
    MM = MarketMaker(config)
    
    if config.starting_epoch:
        MM.load()
        msg = f"\t Resuming {config.name} from epoch {MM.config.starting_epoch}"
        print(msg)
        MM.logger.info(msg)
    if args.plot:
        print(f"Plotting {config.name}")
        MM.plot(plot_book=True, nt=800, wait_time=0.5)
    else:  # do training
        print(f"Training {config.name}")
        MM.train(plot_after=100)
        print(f"Training done! Saved to {config.name}")