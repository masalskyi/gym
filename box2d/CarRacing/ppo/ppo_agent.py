import numpy as np

import torch
import torch.nn as nn
from gym import spaces
from gym.core import ActType

from maslourl.models.continuing.ppo import PPOContinuing

import gym
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, ClipRewardEnv


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CarRacingModel(nn.Module):
    def __init__(self):
        super(CarRacingModel, self).__init__()
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
        self.actor_means = nn.Sequential(layer_init(nn.Linear(512, 2), std=0.01), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.zeros(1, 2))

    def forward(self, x):
        hidden = self.network(x)
        return self.actor_means(hidden), self.critic(hidden)


class Preprocess(gym.ObservationWrapper):
    def __init__(self, env):
        super(Preprocess, self).__init__(env)
        self.observation_space = spaces.Box(0, 255, shape=(46, 96, 3))

    def reset(self, **kwargs):
        return self.observation(self.env.reset())

    def observation(self, observation):
        image = observation[:-50]
        image[np.where((np.logical_and(image >= [101, 203, 101], image <= [101, 230, 101])).all(axis=2))] = np.array(
            [101, 203, 101])
        image[np.where((np.logical_and(image >= [101, 101, 101], image <= [106, 106, 106])).all(axis=2))] = np.array(
            [106, 106, 106])
        return image

class ImageScale(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageScale, self).__init__(env)
        self.observation_space = spaces.Box(0, 1, shape=(84, 84))

    def reset(self, **kwargs):
        return self.observation(self.env.reset())

    def observation(self, observation):
        return observation / 255.0


class ActionChange(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionChange, self).__init__(env)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def action(self, action):
        return np.array([action[0], np.clip(action[1], 0, 1), -np.clip(action[1], -1, 0)])


class CarRacingPPO(PPOContinuing):

    def build_model(self) -> nn.Module:
        return CarRacingModel()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make("CarRacing-v2")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",
                                                   episode_trigger=lambda t: t % capture_every_n_episode == 0)
            env = MaxAndSkipEnv(env, skip=4)
            env = ClipRewardEnv(env)
            env = Preprocess(env)
            env = ActionChange(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = ImageScale(env)
            env = gym.wrappers.FrameStack(env, 4)

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
