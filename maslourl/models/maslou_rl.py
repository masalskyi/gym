from time import time

from maslourl.models.replay_buffer import ReplayBuffer, StateTransition
from maslourl.trackers.file_logger import FileLogger
from maslourl.trackers.average_tracker import AverageRewardTracker
from maslourl.models.utils import calculate_target_values
import numpy as np
from abc import ABC, abstractmethod


# performs double q learning for discrete action space in episode MDP
class MaslouRLModel2QDiscrete(ABC):
    def __init__(self, env, replay_buffer_size=10000):
        self.env = env
        self.model = self.build_model()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        print(self.model.summary())

    def train(self, episodes, max_steps_for_episode, starting_epsilon=0.1, min_epsilon=0.1, epsilon_decay_for_episode=1,
              target_network_replace_frequency_steps=1000, warmup_before_start_trainings_steps=256,
              train_every_x_steps=100,
              training_batch_size=128, discount_factor=0.99,
              model_backup_frequency_episodes=100, path_to_back_up="./",
              episodes_for_average_tracking=100, file_logger=None):
        self.replay_buffer.clear()
        target_model = self.copy_model(self.model)
        average_tracker = AverageRewardTracker(episodes_for_average_tracking)
        epsilon = starting_epsilon
        step_count = 0
        for episode in range(episodes):
            print(f"Starting episode {episode}")
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_for_episode):
                step_count += 1
                action, new_state, reward, done, info = self.step(state, epsilon)
                episode_reward += reward
                if step == max_steps_for_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_for_episode}")
                    done = True
                state_transition = StateTransition(state, action, reward, new_state, done)
                self.replay_buffer.add(state_transition)
                state = new_state
                if step_count % target_network_replace_frequency_steps == 0:
                    # print("Updating target model")
                    target_model = self.copy_model(self.model)

                if self.replay_buffer.length() >= warmup_before_start_trainings_steps and step_count % train_every_x_steps == 0:
                    batch = self.replay_buffer.get_batch(batch_size=training_batch_size)
                    targets = calculate_target_values(self.model, target_model, batch, discount_factor,
                                                      self.env.action_space.n)
                    states = np.array([state_transition.old_state for state_transition in batch])
                    self.train_model(states, targets)
                if done:
                    break
            average_tracker.add(episode_reward)
            average = average_tracker.get_average()
            print(
                f"episode {episode} finished in {step} steps with reward {episode_reward:.2f}. "
                f"Average reward over last {episodes_for_average_tracking}: {average:.2f} "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            if file_logger is not None:
                file_logger.log(episode, step, episode_reward, average)
            if episode != 0 and episode % model_backup_frequency_episodes == 0:
                backup_file = path_to_back_up + f"model_{episode}.h5"
                print(f"Backing up model to {backup_file}")
                self.model.save(backup_file)
            epsilon *= epsilon_decay_for_episode
            epsilon = max(min_epsilon, epsilon)

    def train_model(self, states, targets):
        self.model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)

    def test(self, episodes, max_steps_per_episode, visualize=False):
        rewards = []
        for episode in range(episodes):
            print(f"Starting episode {episode}")
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_per_episode):
                if visualize:
                    self.env.render()
                action, new_state, reward, done, info = self.step(state, 0)
                episode_reward += reward
                if step == max_steps_per_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_per_episode}")
                    done = True
                state = new_state
                if done:
                    break
            print(
                f"episode {episode} finished in {step} steps with reward {episode_reward:.2f}. "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            rewards.append(episode_reward)
        print("Average reward ", np.mean(rewards))


    @abstractmethod
    # returns action that was done, new state, reward, done or not, info
    def step(self, state, epsilon, params):
        raise NotImplementedError("step must be implemented in child")

    @abstractmethod
    def copy_model(self, model):
        raise NotImplementedError("__copy_model must be implemented in child")


    @abstractmethod
    def load_model(self, model_file):
        raise NotImplementedError("load must be implemented in child")

    @abstractmethod
    def build_model(self, model_file):
        raise NotImplementedError("build model must be implemented in child")

    def save_model(self, model_file):
        self.model.save(model_file)