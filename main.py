from MarketMaker.marketmaker import MarketMaker
from MarketMaker.config import get_config, search_for_config
from MarketMaker.util import np, torch, glob

import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load",   help='load a model from a results/name directory')
parser.add_argument("-r", "--resume", action='store_true', help='resume training from a model')
parser.add_argument("-p", "--plot", action='store_true', help='dont train, just plot')

parser.add_argument("--policy", help='use noppo to turn ppo off. use discrete for categorical instead of \
                                      gaussian policy', default='ppo')
parser.add_argument("-ne", "--ne", help="number of epochs to train policy for", type=int)
parser.add_argument("-nb", "--nb", help="number of trajectories to sample for each policy update", type=int)
parser.add_argument("-nt", "--nt", help="number of timesteps to progress trajectory for", type=int)
parser.add_argument("-lr", "--lr", help="learning rate for the policy (Adam) optimizer", type=float)
parser.add_argument("-opt", "--optimizer", help="String name for the optimizer (like Adam, AdamW, SGD... case sensitive)", type=float)
parser.add_argument("--noclip", "-nc", help='dont clip the ratio in PPO', action='store_true', 
                    )
parser.add_argument("--noppo", '-np', help='dont use PPO, just use regular policy grad', action='store_true', 
                    )
parser.add_argument("--noadv",  help='dont use advantages, just returns', action='store_true', 
                    )
parser.add_argument('--uniform', '--nobookquit', '-u', help='dont quit early', action='store_true', 
                    )
parser.add_argument("--no-immediate", "-ni", help='dont use immediate rewards', action='store_true',)
parser.add_argument("--add-time", "-at", help='add time to immediate reward', action='store_true', )
parser.add_argument("--add-inventory", "-ai", help='add inventory to immediate reward', action='store_true', )
parser.add_argument("--always-final", "-af", help='always liquidate at termination', action='store_true', )
parser.add_argument("--plot-after", "-pa", help='plot the market after this many epochs', type=int, default=100)
parser.add_argument("--seed", default=0, type=int)
# TD LAMBDA STUFF?
parser.add_argument("--td", "--TD", "-td", action='store_true')
parser.add_argument("--good-results", help='update plots for all good results', action='store_true')

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
            - args.load
    """
    if args.good_results:   # update the plots for an entire results directory
        results_dir = os.getcwd()+"/good-results"
        for result in glob(results_dir+'/*'):
            print(f"Updating {'/'.join(result.split('/')[-2:])}")
            config, pkl = search_for_config(result)
            config.set_name(config.starting_epoch, save_dir=results_dir)
            MM = MarketMaker(config)
            MM.load()
            MM.plot(nb=1000)
            
    config = get_config(args)
    if not config:
        print("ERROR: No config found... exiting")
        exit()
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
        MM.plot(plot_book=True, nb = 1000, nt=800, wait_time=0.5)
    else:  # do training
        print(f"Training {config.name}")
        MM.train(plot_after=args.plot_after)
        print(f"Training done! Saved to {config.name}")