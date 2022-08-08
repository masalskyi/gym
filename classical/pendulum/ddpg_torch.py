import os
from time import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import random
from maslourl.models.replay_buffer import ReplayBuffer
from maslourl.trackers.average_tracker import AverageRewardTracker
from torch.utils.tensorboard import SummaryWriter


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class DDPG(ABC):

    def __init__(self, cuda=True, seed=1, torch_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic
        temp_env = self.make_env(0, False, 0, "")()
        self.observation_shape = temp_env.observation_space.shape
        self.n_actions = np.prod(temp_env.action_space.shape)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.actor = self.build_actor().to(self.device)
        self.critic = self.build_critic().to(self.device)

    @abstractmethod
    def build_actor(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def build_critic(self) -> torch.nn.Module:
        pass

    def save_agent(self, path, save_2_wandb=False):
        dir_name = "/".join(path.split("/")[:-1])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_name = os.path.basename(path).split(".")[0]
        torch.save(self.actor, os.path.join(os.path.dirname(path), model_name + "_actor.pt"))
        torch.save(self.critic, os.path.join(os.path.dirname(path), model_name + "_critic.pt"))
        if save_2_wandb:
            import wandb
            if not os.path.exists(os.path.join(wandb.run.dir, "models/")):
                os.makedirs(os.path.join(wandb.run.dir, "models/"))
            torch.save(self.actor, os.path.join(wandb.run.dir, "models/best_model_actor.pt"))
            torch.save(self.critic, os.path.join(wandb.run.dir, "models/best_model_critic.pt"))

    def load_agent(self, model_file):
        model_name = os.path.basename(model_file).split(".")[0]
        self.actor = torch.load(self.actor, os.path.join(os.path.dirname(model_file), model_name + "_actor.pt")).to(
            self.device)
        self.critic = torch.load(self.critic, os.path.join(os.path.dirname(model_file), model_name + "_critic.pt")).to(
            self.device)

        self.critic.eval()
        self.actor.eval()

    def update_target_models(self, tau):
        actor_dict = dict(self.actor.named_parameters())
        critic_dict = dict(self.critic.named_parameters())
        actor_target_dict = dict(self.target_actor.named_parameters())
        critic_target_dict = dict(self.target_critic.named_parameters())
        for name in actor_target_dict:
            actor_target_dict[name] = (1 - tau) * actor_target_dict[name].clone() + tau * actor_dict[name].clone()

        for name in critic_target_dict:
            critic_target_dict[name] = (1 - tau) * critic_target_dict[name].clone() + tau * critic_dict[name].clone()
        self.target_actor.load_state_dict(actor_target_dict)
        self.target_critic.load_state_dict(critic_target_dict)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, noise=None):
        self.actor.eval()
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        mu = self.actor(observation)
        if noise is not None:
            mu = mu + torch.tensor(noise(), dtype=torch.float32).to(self.device)
        return mu.cpu().detach().numpy()

    def learn(self, batch_size, discount_factor, tau):
        if self.memory.mem_cntr < batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        self.critic.eval()

        target_actions = self.target_actor(new_state)
        critic_value_ = self.target_critic(new_state, target_actions)
        critic_value = self.critic(state, action)
        critic_value_ *= discount_factor
        not_done = 1 - done
        critic_value_ *= not_done.view(-1, 1)
        target = reward.view(-1, 1) + critic_value_
        self.critic.train()
        self.critic.optimizer.zero_grad()

        critic_loss = torch.nn.functional.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor(state)
        self.actor.train()
        actor_loss = - self.critic(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.update_target_models(tau)

    def train(self, episodes, max_steps_for_episode=1000,
              train_every_step=1, tau=0.005, training_batch_size=64, discount_factor=0.99,
              average_reward_2_save=20, memory_size=100000,
              actor_learning_rate=1e-4, critic_learning_rate=5e-4,
              run_name="DDPG_run_name", verbose=True, ckpt_path="./models/model.pt",
              save_2_wandb=False, seed=1, capture_video=False, capture_every_n_video=20, config=None):

        self.memory = ReplayBuffer(memory_size, self.observation_shape, self.n_actions, discrete=False)
        self.target_actor = self.build_actor().to(self.device)
        self.target_actor.eval()
        self.target_critic = self.build_critic().to(self.device)
        self.target_critic.eval()

        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

        self.noise = OUActionNoise(mu=np.zeros(self.n_actions))
        average_tracker = AverageRewardTracker(average_reward_2_save)

        self.run_name = run_name
        self.writer = SummaryWriter(f"runs/{self.run_name}")
        self.writer.add_text("hyperparameters",
                             "|param|value|\n|-|-|\n%s" % (
                                 "\n".join([f"|{key}|{value}" for key, value in vars(config).items()])))
        global_step = 0
        self.env = self.make_env(seed, capture_video, capture_every_n_video, self.run_name)()
        self.best_reward = -np.inf
        for episode in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_for_episode):
                global_step += 1
                action = self.choose_action(state, noise=self.noise)
                action = np.squeeze(action)
                new_state, reward, done, info = self.env.step(action)
                episode_reward += reward

                self.remember(state, action, reward, new_state, done)
                state = new_state
                if step % train_every_step == 0:
                    self.learn(training_batch_size, discount_factor, tau)
                if done:
                    break
            self.track_info(info, average_tracker, save_2_wandb, verbose, global_step, ckpt_path)

        self.writer.add_scalar("best_avg_reward", self.best_reward)
        self.writer.add_scalar("length_avg_reward", average_reward_2_save)
        self.env.close()
        self.writer.close()

    def track_info(self, info, average_reward_tracker, save_2_wandb, verbose, global_step, ckpt_path):
        for item in info:
            if "episode" in item.keys():
                if verbose:
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                average_reward_tracker.add(item["episode"]["r"])
                avg = average_reward_tracker.get_average()
                if avg > self.best_reward:
                    self.best_reward = avg
                    self.save_agent(ckpt_path, save_2_wandb=save_2_wandb)
                break

    @abstractmethod
    def make_env(self, seed, capture_video, capture_every_n_episode, run_name):
        pass
