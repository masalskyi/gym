import argparse
import os
import time
from distutils.util import strtobool

from dqn_agent import CartPoleDDQ


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment")
    parser.add_argument('--average-reward-tracker', type=int, default=20,
                        help="Tracking the average reward with specified length and save the best model")
    parser.add_argument('--save-best-to-wandb', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="toggle whether to save the best model to w&b")
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument('--verbose', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to verbose on console")
    parser.add_argument('--seed', type=int, default=1, help="seed of the experiment")
    parser.add_argument('--total-episodes', type=int, default=1000, help="total episods of the experiment")
    parser.add_argument('--max-steps-for-episode', type=int, default=1000,
                        help="maximum nubmer of steps per one episod")
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled `torch.backend.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with w&b")
    parser.add_argument('--wandb-project-name', type=str, default="rl-cart-pole", help="the w&b project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent perfomance")
    parser.add_argument('--capture-every-n-video', type=int, default=50, help="capture every nth video")
    parser.add_argument('--target-network-replace', type=int, default=1000, help="steps before updating target network")
    parser.add_argument('--epsilon-start', type=float, default=1, help="the start value of epsilon")
    parser.add_argument('--epsilon-decay', type=float, default=1e-4,
                        help="the additive constant for delaying epsilon every learn step")
    parser.add_argument('--epsilon-min', type=float, default=0.01, help="the minimum value of epsilon")
    parser.add_argument('--batch-size', type=int, default=128, help="the size of a batch for training every learn step")
    parser.add_argument('--memory-size', type=int, default=100000, help="the size of experience buffer")
    parser.add_argument('--gamma', type=float, default=0.99, help="discount gamma")

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

    agent = CartPoleDDQ(args.cuda, args.seed, args.torch_deterministic)
    agent.train(episodes=args.total_episodes,
                max_steps_for_episode=args.max_steps_for_episode,
                starting_epsilon=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay=args.epsilon_decay,
                target_network_replace_frequency_steps=args.target_network_replace,
                training_batch_size=args.batch_size,
                discount_factor=args.gamma,
                episodes_for_average_tracking=args.average_reward_tracker,
                replay_buffer_size=args.memory_size,
                learning_rate=args.learning_rate,
                run_name=run_name,
                verbose=args.verbose,
                save_2_wandb=args.save_best_to_wandb,
                capture_video=args.capture_video,
                capture_every_n_video=args.capture_every_n_video,
                config=args)