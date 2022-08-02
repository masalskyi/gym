import argparse
import os
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)
from maslourl.models.ppo_torch import MaslouTorchAgent, MaslouPPO
import gym


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CarRacerTorchAgent(MaslouTorchAgent):
    def __init__(self, envs: gym.vector.VectorEnv):
        super(CarRacerTorchAgent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1.)
        self.actor_means = layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        action_mean = self.actor_means(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)

    def get_action_greedy(self, x):
        hidden = self.network(x / 255.0)
        action_mean = self.actor(hidden)
        return action_mean


class CarRacerPPO(MaslouPPO):
    def __init__(self, args):
        super(CarRacerPPO, self).__init__(args, discrete=False)
        print(self.agent)

    def build_agent(self) -> MaslouTorchAgent:
        return CarRacerTorchAgent(self.envs)

    def make_env(self, gym_id, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make(gym_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                                   episode_trigger=lambda t: t % capture_every_n_episode == 0)
            env = MaxAndSkipEnv(env, skip=4)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)

            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment")
    parser.add_argument('--gym-id', type=str, default="CarRacing-v2",
                        help='the id of the openai gym environment')
    parser.add_argument('--verbose', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to verbose")
    parser.add_argument('--average-reward-2-save', type=int, default=20,
                        help="Tracking the average reward with specified length and save the best model")
    parser.add_argument('--save-best-to-wandb', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to save the best model to w&b")
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument('--seed', type=int, default=1, help="seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=10000000, help="total timesteps of the experiment")
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled `torch.backend.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with w&b")
    parser.add_argument('--wandb-project-name', type=str, default="rl-ppo-carracing", help="the w&b project name")
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
    breakout_ppo = CarRacerPPO(args)
    breakout_ppo.train()
