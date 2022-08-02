from maslourl.models.ppo_torch import MaslouTorchModel, MaslouPPOAgentEval
import gym
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv
)
import argparse
import os
from distutils.util import strtobool
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class BreakoutTorchAgent(MaslouTorchModel):
    def __init__(self, env):
        super(BreakoutTorchAgent, self).__init__()
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
        self.actor = layer_init(nn.Linear(512, env.action_space.n), std=0.01)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_action_greedy(self, x):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        return torch.argmax(logits, dim=1)

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(84, 84, 1), dtype=np.uint8)
    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = frame.reshape((84, 84, 1))
        return new_frame

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def create_env():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)

    env.seed(21)
    env.action_space.seed(21)
    env.observation_space.seed(21)
    return env


class BreakoutPPO(MaslouPPOAgentEval):
    def __init__(self, env):
        super(BreakoutPPO, self).__init__(use_cuda=True, env=env)

    def build_agent(self) -> MaslouTorchModel:
        return BreakoutTorchAgent(self.env)

env = create_env()

agent = BreakoutPPO(env)
agent.load_agent("./model/best_model.pt")
episodes = 1
import cv2
for episode in range(episodes):
    reward = 0
    done = False
    obs = agent.env.reset()
    print(obs.shape)
    while not done:
        action = agent.agent.get_action_greedy(torch.tensor([obs]).to("cuda", dtype=torch.float))
        # print(action)
        obs, r, done, _ = agent.env.step(action)
        reward += r
        img = agent.env.render(mode="rgb_array")
        cv2.imshow("test", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print(f"Episode {episode + 1} end with reward {reward}")
