import time

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow_probability.python.distributions.categorical import Categorical
from abc import ABC, abstractmethod
import random
import numpy as np
import os
import gym

class MaslouRLModelPPODiscrete(ABC):
    def __init__(self):
        self.env = self.make_env(0, 0, False, 0, "temp")()
        self.input_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.feature_extractor = self.build_feature_extractor()
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        self.actor.summary()
        self.critic.summary()
        self.feature_extractor.summary()

    @abstractmethod
    def make_env(self, seed, idx, capture_video, capture_every_n_video, run_name):
        pass

    @abstractmethod
    def build_actor(self) -> keras.Model:
        pass

    @abstractmethod
    def build_feature_extractor(self) -> keras.Model:
        pass

    @abstractmethod
    def build_critic(self) -> keras.Model:
        pass

    def get_action_and_value(self, x, action=None):
        hidden = self.feature_extractor.predict(x, verbose=0)
        probs = self.actor.predict(hidden, verbose=0)
        distribution = Categorical(probs=probs)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic.predict(hidden, verbose=0)

    def train(self, learning_rate=2.5e-4, num_steps=128,
              num_envs=4, seed=42, capture_video=True,
              capture_every_n_video=50, run_name="PPO_run_name",
              total_timesteps=1000000, anneal_lr=True):
        batch_size = num_steps * num_envs
        envs = gym.vector.SyncVectorEnv([self.make_env(seed + i,
                                                       i,
                                                       capture_video,
                                                       capture_every_n_video,
                                                       run_name) for i in range(num_envs)])

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["TF_CUDNN_DETERMINISTIC"] = '1'

        optimizer = Adam(lr=learning_rate, epsilon=1e-5)
        obs = np.zeros((num_steps, num_envs) + envs.single_observation_space.shape, dtype=float)
        actions = np.zeros((num_steps, num_envs) + envs.single_action_space.shape, dtype=float)
        logprobs = np.zeros((num_steps, num_envs), dtype=float)
        rewards = np.zeros((num_steps, num_envs), dtype=float)
        dones = np.zeros((num_steps, num_envs), dtype=float)
        values = np.zeros((num_steps, num_envs), dtype=float)

        global_step = 0
        start_time = time.time()
        next_obs = envs.reset()
        next_done = np.zeros(num_envs)
        num_updates = total_timesteps // batch_size

        for update in range(1, num_updates + 1):
            if anneal_lr:
                frac = 1.0 - (update - 1) / num_updates
                lrnow = frac * learning_rate
                optimizer._lr = lrnow

            for step in range(0, num_steps):
                global_step += num_envs
                obs[step] = next_obs
                dones[step] = next_done

                action, logprob, entropy, value = self.get_action_and_value(next_obs)
                values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, done, info = envs.step(action)
                rewards[step] = reward
                print(info)
                # for item in info:
                #     if "episode" in item.keys():
                #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                #         break
