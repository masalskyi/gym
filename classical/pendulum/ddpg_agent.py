import torch
import gym
from maslourl.models.continuing.ddpg import DDPG
import numpy as np
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PendulumCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(PendulumCritic, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim + n_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.value_function = nn.Linear(300, 1)

    def forward(self, state, action):
        state_action = torch.hstack((state, action))
        features = self.feature(state_action)
        return self.value_function(features)

class PendulumActor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(PendulumActor, self).__init__()
        self.state_feature = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.action = nn.Sequential(
            nn.Linear(300, n_actions),
            nn.Tanh()
        )

    def forward(self, state):
        state_feature = self.state_feature(state)
        return self.action(state_feature) * 2

class PendulumDDPG(DDPG):
    def make_env(self, seed, capture_video, capture_every_n_episode, run_name):
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

    def build_actor(self) -> torch.nn.Module:
        return PendulumActor(self.observation_shape[0], self.n_actions)

    def build_critic(self) -> torch.nn.Module:
        return PendulumCritic(self.observation_shape[0], self.n_actions)


    def track_info(self, info, average_reward_tracker, save_2_wandb, verbose, global_step, ckpt_path):
        if "episode" in info.keys():
            item = info["episode"]
            if verbose:
                print(f"global_step={global_step}, episodic_return={item['r']}")
            self.writer.add_scalar("charts/episodic_return", item["r"], global_step)
            self.writer.add_scalar("charts/episodic_length", item["l"], global_step)
            average_reward_tracker.add(item["r"])
            avg = average_reward_tracker.get_average()
            if avg > self.best_reward:
                self.best_reward = avg
                self.save_agent(ckpt_path, save_2_wandb=save_2_wandb)
            return