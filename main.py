import torch, numpy as np, argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name", required=True, type=str, choices=["cartpole", "pendulum", "cheetah"]
)
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true")
parser.add_argument("--seed", type=int, default=1)

parser.set_defaults(use_baseline=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    #TODO: once models are complete do this shiiit
    
    # train model
    model = PolicyGradient(env, config, args.seed) if not args.ppo else PPO(env, config, args.seed)
    model.run()
