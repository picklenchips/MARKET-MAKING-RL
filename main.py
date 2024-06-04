from marketmaker import MarketMaker
import argparse
from config import Config, get_config
import numpy as np
import torch
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default='ppo')
parser.add_argument("-ne", "--ne", type=int)
parser.add_argument("-nb", "--nb", type=int)
parser.add_argument("-nt", "--nt", type=int)
parser.add_argument("-l", "--load",   help='load a model from a filepath/name & resume')
parser.add_argument("-r", "--resume", action='store_true', help='resume training from a model')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noclip", action='store_true')
parser.add_argument("-p", "--plot", action='store_true', help='dont train, just plot')
# TD LAMBDA STUFF?
parser.add_argument("--td", action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = False
    if args.load:  # load a model from a filepath/name & resume
        config = get_config(args.load)
    if not config:  # DEFAULT CONFIG
        if args.td: 
            config = Config(trajectory='TD')
        else:
            config = Config()
    
    # if we add other policies LOL
    policy = args.policy.lower()
    if policy != 'ppo': 
        print('as if we have implemented another policy...')
        pass
    if args.nb: config.nb = args.nb
    if args.nt: config.nt = args.nt
    if args.ne: config.ne = args.ne
    if args.noclip: 
        config.do_clip = False
    
    config.set_name(make_new=(not args.resume and not args.plot and not args.load))
    if args.resume or args.plot:  # resume from same specifications as given above
        # assume there is only one thing in each directory
        filename = glob(config.save_dir+"*.pth")[-1]
        config = get_config(filename)

    MM = MarketMaker(config)
    if args.resume or args.load or args.plot: 
        MM.load()
        print(f"\t Resuming {config.name} from epoch {MM.config.starting_epoch}")
    if args.plot:
        print(f"Plotting {config.name}")
        MM.plot()
    else:  # do training
        print(f"Training {config.name}")
        MM.train()
        print(f"Training done! Saved to {config.name}")