import numpy as np

import torch
import torch.nn as nn
from gym import spaces

from maslourl.models.continuing.ppo import PPOContinuing

import gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class BipedalWalkerModel(nn.Module):
    def __init__(self):
        super(BipedalWalkerModel, self).__init__()
        observations = 24
        n_actions = 4
        self.network = nn.Sequential(
            layer_init(nn.Linear(observations, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1.)
        self.actor_means = nn.Sequential(layer_init(nn.Linear(512, n_actions), std=0.01), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, x):
        hidden = self.network(x)
        return self.actor_means(hidden), self.critic(hidden)


class BipedalWalkerPPO(PPOContinuing):

    def build_model(self) -> nn.Module:
        return BipedalWalkerModel()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make("BipedalWalker-v3")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                                   episode_trigger=lambda t: t % capture_every_n_episode == 0)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    def track_info(self, info, average_reward_tracker, save_2_wandb, verbose, global_step, ckpt_path):
        if "episode" in info.keys():
            for item in info["episode"]:
                if item is not None:
                    if verbose:
                        print(f"global_step={global_step}, episodic_return={item['r']}")
                    self.writer.add_scalar("charts/episodic_return", item["r"], global_step)
                    self.writer.add_scalar("charts/episodic_length", item["l"], global_step)
                    average_reward_tracker.add(item["r"])
                    avg = average_reward_tracker.get_average()
                    if avg > self.best_reward:
                        self.best_reward = avg
                        self.save_agent(ckpt_path, save_2_wandb=save_2_wandb)
                    break
