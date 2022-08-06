from maslourl.models.ddqn import DDQDiscrete
import gym
import torch
from torch import nn
class CartPoleTorchModel(torch.nn.Module):
    def __init__(self):
        super(CartPoleTorchModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.Adv = nn.Linear(64, 2)
        self.Val = nn.Linear(64, 1)
    def forward(self, state):
        hidden = self.network(state)
        adv = self.Adv(hidden)
        result = torch.add(self.Val(hidden), adv - adv.mean())
        return result

class CartPoleDDQ(DDQDiscrete):
    def build_env(self, seed, capture_video, capture_every_n_video) -> gym.Env:
        env = gym.make("CartPole-v1")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, "./videos/" + self.run_name, lambda x: x % capture_every_n_video == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    def build_model(self) -> torch.nn.Module:
        return CartPoleTorchModel()
