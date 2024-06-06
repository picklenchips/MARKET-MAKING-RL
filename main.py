from marketmaker import MarketMaker, UniformMarketMaker
import argparse
from config import get_config, Config
import numpy as np
import torch
from config import SAVEDIR

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load",   help='load a model from a results/name directory')
parser.add_argument("-r", "--resume", action='store_true', help='resume training from a model')
parser.add_argument("-p", "--plot", action='store_true', help='dont train, just plot')

parser.add_argument("--policy", default='ppo')
parser.add_argument("-ne", "--ne", type=int)
parser.add_argument("-nb", "--nb", type=int)
parser.add_argument("-nt", "--nt", type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noclip", action='store_true', help='dont clip the ratio in PPO')
parser.add_argument("--noppo", action='store_true', help='dont use PPO, just use regular policy grad')
parser.add_argument("--noadv", action='store_true', help='dont use advantages, just returns')
parser.add_argument( '--nobookquit','--uniform', action='store_true', help='dont quit early')
parser.add_argument("--immediate", action='store_true', help='use immediate rewards')
parser.add_argument("--subtract_time", "-st", action='store_true', help='subtract time from reward')

# TD LAMBDA STUFF?
parser.add_argument("--td", action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    #torch.random.manual_seed(args.seed)
    #np.random.seed(args.seed)

    config = get_config(args)
    #print(config)

    MM = MarketMaker(config) if config.book_quit else UniformMarketMaker(config)
    if config.starting_epoch:
        MM.load()
        print(f"\t Resuming {config.name} from epoch {MM.config.starting_epoch}")
    if args.plot:
        print(f"Plotting {config.name}")
        MM.plot(plot_book=True, nt=800, wait_time=0.5)
    else:  # do training
        print(f"Training {config.name}")
        MM.train(plot_after=100)
        print(f"Training done! Saved to {config.name}")