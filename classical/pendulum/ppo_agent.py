import numpy as np

import torch
import torch.nn as nn

from maslourl.models.continuing.ppo import PPOContinue

import gym


def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PendulumTorchAgent(nn.Module):
    def __init__(self):
        super(PendulumTorchAgent, self).__init__()
        obs_size = 3
        n_actions = 1
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_size, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 64)),
            nn.ReLU()
        )
        self.critic = layer_init(nn.Linear(64, 1), std=1.)
        self.actor = layer_init(nn.Linear(64, n_actions), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_actions))

    def forward(self, x):
        hidden = self.network(x)
        return self.actor(hidden), self.critic(hidden)


class PendulumPPO(PPOContinue):

    def build_model(self) -> nn.Module:
        return PendulumTorchAgent()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make('Pendulum-v1', g=9.81)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            if capture_video:
                env = gym.wrappers.RecordVideo(env, "./videos/" + run_name, lambda x: x % capture_every_n_episode == 0)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    def track_info(self, info, average_reward_tracker, save_2_wandb, verbose, global_step, ckpt_path):
        if "episode" in info.keys():
            for item in info["episode"]:
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
