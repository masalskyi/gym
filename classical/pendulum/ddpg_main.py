import argparse
import os
import time
from distutils.util import strtobool

from ddpg_agent import PendulumDDPG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment")
    parser.add_argument('--average-reward-2-save', type=int, default=20,
                        help="Tracking the average reward with specified length and save the best model")
    parser.add_argument('--save-best-to-wandb', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to save the best model to w&b")
    parser.add_argument('--critic-learning-rate', type=float, default=2.5e-4,
                        help="the learning rate of the critic optimizer")
    parser.add_argument('--actor-learning-rate', type=float, default=1e-4,
                        help="the learning rate of the actor optimizer")
    parser.add_argument('--verbose', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to verbose on console")
    parser.add_argument('--seed', type=int, default=1, help="seed of the experiment")
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
    parser.add_argument('--total-episodes', type=int, default=1000, help="the number of episodes to run")
    parser.add_argument('--max-steps', type=int, default=1000, help="the maximum number of steps for one episode")
    parser.add_argument('--gamma', type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument('--train-every-n-step', type=int, default=1, help="perform learning every n step")
    parser.add_argument('--tau', type=float, default=0.005, help="the strength of shifting model parameters to target")
    parser.add_argument('--train-batch-size', type=int, default=64, help="the size of a batch for one learning step")
    parser.add_argument('--memory-size', type=int, default=1000000, help="the size of experience replay buffer")

    args = parser.parse_args()
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
    pendulum_ddpg = PendulumDDPG(cuda=args.cuda, seed=args.seed, torch_deterministic=args.torch_deterministic)
    pendulum_ddpg.train(episodes=args.total_episodes, max_steps_for_episode=args.max_steps,
                        train_every_step=args.train_every_n_step, tau=args.tau,
                        training_batch_size=args.train_batch_size, discount_factor=args.gamma,
                        average_reward_2_save=args.average_reward_2_save, memory_size=args.memory_size,
                        actor_learning_rate=args.actor_learning_rate, critic_learning_rate=args.critic_learning_rate,
                        run_name=run_name, verbose=args.verbose, save_2_wandb=args.track, seed=args.seed,
                        capture_video=args.capture_video, capture_every_n_video=args.capture_every_n_video, config=args)
