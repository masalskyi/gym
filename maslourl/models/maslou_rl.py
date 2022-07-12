from time import time
from abc import ABC, abstractmethod
from tensorflow.keras.models import load_model
import numpy as np

from maslourl.models.replay_buffer import ReplayBuffer
from maslourl.trackers.file_logger import FileLogger
from maslourl.trackers.average_tracker import AverageRewardTracker
from maslourl.models.utils import get_q_values, select_action_epsilon_greedy


# performs double q learning for discrete action space in episode MDP
class MaslouRLModel2QDiscrete(ABC):
    def __init__(self, env, replay_buffer_size=10000):
        self.env = env
        self.Q_eval = self.build_model()
        self.Q_target = self.build_model()
        self.update_target_model()
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.action_space = np.array([i for i in range(self.n_actions)])
        self.memory = ReplayBuffer(replay_buffer_size, input_shape=self.input_shape, n_actions=self.n_actions)

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        print(self.Q_eval.summary())

    def train(self, episodes, max_steps_for_episode, starting_epsilon=1, epsilon_min=0.01, epsilon_decay=0.996,
              target_network_replace_frequency_steps=1000,
              training_batch_size=128, discount_factor=0.99,
              model_backup_frequency_episodes=100, path_to_back_up="./",
              episodes_for_average_tracking=100, file_logger=None):
        self.memory.clear()
        average_tracker = AverageRewardTracker(episodes_for_average_tracking)
        self.epsilon = starting_epsilon
        self.target_network_replace_frequency_steps = target_network_replace_frequency_steps
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        for episode in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_for_episode):
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                if step == max_steps_for_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_for_episode}")
                    done = True
                self.remember(state, action, reward, new_state, done)
                state = new_state
                self.learn(training_batch_size, discount_factor)
                if done:
                    break
            average_tracker.add(episode_reward)
            average = average_tracker.get_average()
            print(
                f"episode {episode} finished in {step} steps with reward {episode_reward:.2f}. "
                f"Average reward over last {episodes_for_average_tracking}: {average:.2f} "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            if file_logger is not None:
                file_logger.log(episode, step, episode_reward, average, self.epsilon)
            if episode != 0 and episode % model_backup_frequency_episodes == 0:
                backup_file = path_to_back_up + f"model_{episode}.h5"
                print(f"Backing up model to {backup_file}")
                self.model.save(backup_file)

    def test(self, episodes, max_steps_per_episode, visualize=False):
        rewards = []
        self.epsilon = 0
        for episode in range(episodes):
            print(f"Starting episode {episode}")
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_per_episode):
                if visualize:
                    self.env.render()
                action = self.choose_action(state)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                if step == max_steps_per_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_per_episode}")
                    done = True
                if done:
                    break
            print(
                f"episode {episode} finished in {step} steps with reward {episode_reward:.2f}. "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            rewards.append(episode_reward)
        print("Average reward ", np.mean(rewards))

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        q_values = get_q_values(self.Q_eval, state)
        action = select_action_epsilon_greedy(q_values, self.epsilon)
        return action

    def learn(self, batch_size, discount_factor):
        if self.memory.mem_cntr > batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)
            action_indices = np.dot(action, self.action_space)

            q_eval = self.Q_eval.predict(new_state, verbose=0)
            q_target = self.Q_target.predict(new_state, verbose=0)

            q_pred = self.Q_eval.predict(state, verbose=0)
            max_actions = np.argmax(q_eval, axis=1)

            q_y = q_pred
            batch_indices = np.arange(batch_size, dtype=np.int32)
            q_y[batch_indices, action_indices] = reward + discount_factor * \
                                                 q_target[batch_indices, max_actions.astype(int)] * done
            _ = self.Q_eval.fit(state, q_y, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.memory.mem_cntr % self.target_network_replace_frequency_steps:
                self.update_target_model()

    def update_target_model(self):
        self.Q_target.set_weights(self.Q_eval.get_weights())

    def load_model(self, model_file, prepare_target=False):
        self.Q_eval = load_model(model_file)
        if prepare_target:
            self.Q_target = self.build_model()
            self.update_target_model()

    @abstractmethod
    def build_model(self):
        raise NotImplementedError("build model must be implemented in child")

    def save_model(self, model_file):
        self.model.save(model_file)
