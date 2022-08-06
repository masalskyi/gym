import numpy as np

import torch
import torch.nn as nn

from maslourl.models.ppo import PPODiscrete

import gym

def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class BreakoutTorchAgent(nn.Module):
    def __init__(self):
        super(BreakoutTorchAgent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU()
        )
        self.critic = layer_init(nn.Linear(64, 1), std=1.)
        self.actor = nn.Sequential(layer_init(nn.Linear(64, 2), std=0.01), nn.Softmax())

    def forward(self, x):
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)


class CartPolePPO(PPODiscrete):

    def build_model(self) -> nn.Module:
        return BreakoutTorchAgent()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make("CartPole-v1")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk
