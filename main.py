from marketmaker import MarketMaker
import argparse
from config import Config, get_config
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", default='ppo')
parser.add_argument("-ne", "--ne", type=int)
parser.add_argument("-nb", "--nb", type=int)
parser.add_argument("-nt", "--nt", type=int)
parser.add_argument("-l", "--load", help='load a model from a filepath/name')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--noclip", action='store_true')
# TD LAMBDA STUFF?
parser.add_argument("--td", action='store_true')

if __name__ == "__main__":
    args = parser.parse_args()
    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = False
    if args.load:
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
    if args.noclip: config.do_clip = False

    MM = MarketMaker(config)
    print(f"Training {config.name}")
    MM.train()
    print(f"Training done! Saved to {config.name}")