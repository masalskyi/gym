import argparse
import os
import time
from distutils.util import strtobool

from ppo_agent import PendulumPPO
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment")
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='the id of the openai gym environment')
    parser.add_argument('--average-reward-2-save', type=int, default=20,
                        help="Tracking the average reward with specified length and save the best model")
    parser.add_argument('--save-best-to-wandb', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to save the best model to w&b")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument('--verbose', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to verbose on console")
    parser.add_argument('--seed', type=int, default=1, help="seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=10000000, help="total timesteps of the experiment")
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled `torch.backend.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with w&b")
    parser.add_argument('--wandb-project-name', type=str, default="rl-pendulum", help="the w&b project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent perfomance")
    parser.add_argument('--capture-every-n-video', type=int, default=50, help="capture every nth video")
    parser.add_argument('--num-envs', type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument('--num-steps', type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="use General Andvantage estimation for advantage computation")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help="the lambda for GAE")
    parser.add_argument('--num-minibutches', type=int, default=4,
                        help="the number of minibatches")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--norm_adv', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coeficient")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggles whether or not to use a cliped loss for the value function as per the paper")
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient for entropy loss")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient for loss of value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="the maximum norm of gradient clipping")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibutches)
    return args


if __name__ == '__main__':
    args = parse_args()
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(project=args.wandb_project_name,
                   entity=args.wandb_entity,
                   sync_tensorboard=True,
                   config=vars(args),
                   name=run_name,
                   monitor_gym=True,
                   save_code=True)
    cart_pole_ppo = PendulumPPO(cuda=args.cuda, seed=args.seed, torch_deterministic=args.torch_deterministic)
    cart_pole_ppo.train(learning_rate=args.learning_rate, num_steps=args.num_steps,
                        num_envs=args.num_envs, seed=args.seed,
                        capture_video=args.capture_video, capture_every_n_video=args.capture_every_n_video, run_name=run_name,
                        total_timesteps=args.total_timesteps, anneal_lr=args.anneal_lr, gae=args.gae, discount_gamma=args.gamma,
                        gae_lambda=args.gae_lambda, update_epochs=args.update_epochs,
                        minibatches=args.num_minibutches, norm_adv=args.norm_adv, clip_coef=args.clip_coef,
                        clip_vloss=args.clip_vloss, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                        save_2_wandb=args.track, config=args)

