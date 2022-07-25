from time import time
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras.losses

from tensorflow.keras.models import load_model
import numpy as np

from maslourl.models.replay_buffer import ReplayBuffer
from maslourl.trackers.file_logger import FileLogger
from maslourl.trackers.average_tracker import AverageRewardTracker
from maslourl.models.utils import get_q_values, select_action_epsilon_greedy, select_best_action


# performs double q learning for discrete action space in episode MDP
class MaslouRLModel2QDiscrete(ABC):
    def __init__(self, env, replay_buffer_size=10000, train_state=True, need_to_initiate_memory=True):
        self.env = env
        self.input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.train_state = train_state
        self.memory = None
        if self.train_state and need_to_initiate_memory:
            self.memory = ReplayBuffer(replay_buffer_size, input_shape=self.input_shape, n_actions=self.n_actions)

        self.Q_eval = self.build_model()
        if self.train_state:
            self.Q_target = self.build_model()
            self.update_target_model()
        self.epsilon = 0

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        print(self.Q_eval.summary())

    def train(self, episodes, max_steps_for_episode=1000, starting_epsilon=1, epsilon_min=0.01, epsilon_decay=0.996,
              target_network_replace_frequency_steps=1000,
              training_batch_size=128, discount_factor=0.99,
              model_backup_frequency_episodes=100, path_to_back_up="./",
              episodes_for_average_tracking=100, file_logger=None):
        if not self.train_state:
            raise ValueError("Agent is not in train state")
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
                self.remember(state, action, reward, new_state, done)
                state = new_state
                self.learn(training_batch_size, discount_factor)
                if done:
                    break
            average_tracker.add(episode_reward)
            average = average_tracker.get_average()
            print(
                f"episode {episode} finished in {step + 1} steps with reward {episode_reward:.2f}. "
                f"Average reward over last {episodes_for_average_tracking}: {average:.2f} "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            if file_logger is not None:
                file_logger.log(episode, step + 1, episode_reward, average, self.epsilon)
            if episode != 0 and episode % model_backup_frequency_episodes == 0:
                backup_file = path_to_back_up + f"model_{episode}.h5"
                print(f"Backing up model to {backup_file}")
                self.save_model(backup_file)
        self.epsilon = 0

    def test(self, episodes, max_steps_per_episode, visualize=False):
        if self.train_state:
            raise ValueError("Model is in train state")
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
                action = self.choose_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                if step == max_steps_per_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_per_episode}")
                    done = True
                if done:
                    self.env.render()
                    break
            print(
                f"episode {episode} finished in {step + 1} steps with reward {episode_reward:.2f}. "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            rewards.append(episode_reward)
        print("Average reward ", np.mean(rewards))

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state):
        q_values = get_q_values(self.Q_eval, state)
        if self.train_state:
            action = select_action_epsilon_greedy(q_values, self.epsilon)
        else:
            action = select_best_action(q_values)
        return action

    def learn(self, batch_size, discount_factor):
        if self.memory.mem_cntr > batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)
            action_indices = action

            q_eval = self.Q_eval.predict(new_state, verbose=0)
            q_target = self.Q_target.predict(new_state, verbose=0)

            q_pred = self.Q_eval.predict(state, verbose=0)
            max_actions = np.argmax(q_eval, axis=1)

            q_y = q_pred[:]
            batch_indices = np.arange(batch_size, dtype=np.int32)
            q_y[batch_indices, action_indices] = reward + discount_factor * \
                                                 q_target[batch_indices, max_actions.astype(int)] * (1 - done)
            _ = self.Q_eval.fit(state, q_y, verbose=0)

            self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.memory.mem_cntr % self.target_network_replace_frequency_steps == 0:
                self.update_target_model()

    def update_target_model(self):
        self.Q_target.set_weights(self.Q_eval.get_weights())

    def load_model(self, model_file):
        path = model_file.split(".")
        path_without_ext = ".".join(path[:-1])
        self.Q_eval = load_model(path_without_ext + "." + path[-1])
        if self.train_state:
            self.Q_target = load_model(path_without_ext + "_t." + path[-1])
            self.memory = ReplayBuffer.load(path_without_ext + "_memory.npz")

    @abstractmethod
    def build_model(self):
        raise NotImplementedError("build model must be implemented in child")

    def save_model(self, model_file):
        path = model_file.split(".")
        path_without_ext = ".".join(path[:-1])
        self.Q_eval.save(path_without_ext + "." + path[-1])
        if self.train_state:
            self.Q_target.save(path_without_ext + "_t." + path[-1])
            self.memory.save(path_without_ext + "_memory.npz")


class MaslouRLModelDDPGContinuous(ABC):
    def __init__(self, env, replay_buffer_size=10000, train_state=True, need_to_initiate_memory=True):
        self.env = env
        self.input_shape = env.observation_space.shape
        self.n_actions = np.prod(env.action_space.shape)
        self.min_actions = env.action_space.low
        self.max_actions = env.action_space.high
        self.train_state = train_state
        self.memory = None
        if self.train_state and need_to_initiate_memory:
            self.memory = ReplayBuffer(replay_buffer_size, input_shape=self.input_shape, n_actions=self.n_actions,
                                       discrete=False)
        self.actor = self.build_actor_model()
        self.critic = self.build_critic_model()

        if self.train_state:
            self.actor_target = self.build_actor_model()
            self.critic_target = self.build_critic_model()
            self.update_target_models(tau=1)

    def summary(self):
        print("Observation space:", self.env.observation_space)
        print("Action space:", self.env.action_space)
        print(self.actor.summary())
        print(self.critic.summary())

    @abstractmethod
    def build_actor_model(self) -> tensorflow.keras.models.Model:
        pass

    @abstractmethod
    def build_critic_model(self) -> tensorflow.keras.models.Model:
        pass

    def save_models(self, model_file):
        path = model_file.split(".")
        path_without_ext = ".".join(path[:-1])
        self.actor.save(path_without_ext + "_actor." + path[-1])
        self.critic.save(path_without_ext + "_critic." + path[-1])

        if self.train_state:
            self.actor_target.save(path_without_ext + "_actor_t." + path[-1])
            self.critic_target.save(path_without_ext + "_critic_t." + path[-1])
            self.memory.save(path_without_ext + "_memory.npz")

    def load_model(self, model_file):
        path = model_file.split(".")
        path_without_ext = ".".join(path[:-1])
        actor_model_path = path_without_ext + "_actor.h5"
        critic_model_path = path_without_ext + "_critic.h5"
        actor_t_model_path = path_without_ext + "_actor_t.h5"
        critic_t_model_path = path_without_ext + "_critic_t.h5"
        memory_path = path_without_ext+"_memory.npz"
        self.actor = load_model(actor_model_path)
        self.critic = load_model(critic_model_path)
        if self.train_state:
            self.actor_target = load_model(actor_t_model_path)
            self.critic_target = load_model(critic_t_model_path)
            self.memory = ReplayBuffer.load(memory_path)

    def update_target_models(self, tau):
        weights = []
        targets = self.actor_target.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + (1 - tau) * targets[i])
        self.actor_target.set_weights(weights)

        weights = []
        targets = self.critic_target.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + (1 - tau) * targets[i])
        self.critic_target.set_weights(weights)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, state, noise=None):
        state = state.reshape((-1, *self.input_shape))
        actions = self.actor.predict(state, verbose=0)
        if self.train_state:
            actions += np.random.normal(0, noise, self.n_actions)
            actions = np.clip(actions, self.min_actions, self.max_actions)
        return actions

    def learn(self, batch_size, discount_factor, tau):
        if self.memory.mem_cntr < batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.actor_target(states_)
            critic_values_ = tf.squeeze(self.critic_target([states_, target_actions]), axis=1)
            critic_values = tf.squeeze(self.critic([states, actions]), axis=1)
            target = rewards + discount_factor * (1 - done) * critic_values_
            critic_loss = tensorflow.keras.losses.MSE(target, critic_values)
            critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic([states, new_policy_actions])
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        self.update_target_models(tau)

    def train(self, episodes, max_steps_for_episode=1000, train_every_step=1, tau=0.005, noise=0.1,
              training_batch_size=64, discount_factor=0.99,
              model_backup_frequency_episodes=100, path_to_back_up="./",
              episodes_for_average_tracking=100, file_logger=None):
        if not self.train_state:
            raise ValueError("Agent is not in train state")
        self.memory.clear()
        average_tracker = AverageRewardTracker(episodes_for_average_tracking)
        for episode in range(episodes):
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_for_episode):
                action = self.choose_action(state, noise=noise)
                action = np.squeeze(action)
                new_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                if step == max_steps_for_episode:
                    print(f"Episode reached the maximum number of steps. {max_steps_for_episode}")
                    done = True
                self.remember(state, action, reward, new_state, done)
                state = new_state
                if step % train_every_step == 0:
                    self.learn(training_batch_size, discount_factor, tau)
                if done:
                    break
            average_tracker.add(episode_reward)
            average = average_tracker.get_average()
            print(
                f"episode {episode} finished in {step + (1 if done else 0)} steps with reward {episode_reward:.2f}. "
                f"Average reward over last {episodes_for_average_tracking}: {average:.2f} "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            if file_logger is not None:
                file_logger.log(episode, step + 1, episode_reward, average, 0)
            if episode != 0 and episode % model_backup_frequency_episodes == 0:
                backup_file = path_to_back_up + f"model_{episode}.h5"
                print(f"Backing up model to {backup_file}")
                self.save_models(backup_file)

    def test(self, episodes, max_steps_per_episode=1000, visualize=False):
        rewards = []
        if self.train_state:
            raise ValueError("Agent is in train state")
        for episode in range(episodes):
            print(f"Starting episode {episode}")
            episode_reward = 0
            state = self.env.reset()
            episode_start_time = time()
            for step in range(max_steps_per_episode):
                if visualize:
                    self.env.render()
                action = self.choose_action(state)
                action = np.squeeze(action)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

                if done:
                    self.env.render()
                    break
            print(
                f"episode {episode} finished in {step} steps with reward {np.array([episode_reward])[0]:.2f}. "
                f"And took: {(time() - episode_start_time):.2f} seconds. ")
            rewards.append(episode_reward)
        print("Average reward ", np.mean(rewards))
