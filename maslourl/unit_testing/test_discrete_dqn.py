import unittest

import cv2
import numpy as np
import random
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from utils.ping_pong_env import PingPongEnv
from utils.ping_pong_model import PingPongModel


class PongEnvTest(unittest.TestCase):
    weights = None
    load_data = None
    agent = None
    epsilon_history = None

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        random.seed(0)
        tf.random.set_seed(0)
        cls.load_data = np.load("./data/unit1.npz", allow_pickle=True)
        print(list(cls.load_data.keys()))

        replay_buffer_size = 1200
        training_batch_size = 32
        minimum_epsilon = 0.02
        epsilon_decay = 1e-5
        discount_factor = 0.99

        env = PingPongEnv(slide_window_length=4, image_resize=(80, 80), skip_steps=4)
        cls.agent = PingPongModel(env, replay_buffer_size=replay_buffer_size)
        cls.agent.epsilon_min = minimum_epsilon
        cls.agent.epsilon_decay = epsilon_decay
        cls.agent.target_network_replace_frequency_steps = 1000
        cls.agent.epsilon = 1
        cls.weights = []
        n_steps = 0
        cls.epsilon_history = []
        done = False
        observation = env.reset()
        all = 764
        for i in range(all + 1):
            if i % 100 == 0:
                print(f"{i} / {all}")
            # action = cls.agent.choose_action(observation)
            observation_, reward, done, info = env.step(0)
            n_steps += 1
            cls.agent.remember(observation, 0,
                               reward, observation_, int(done))
            cls.agent.learn(training_batch_size, discount_factor)
            cls.weights.append(cls.agent.Q_eval.get_weights())
            observation = observation_
            cls.epsilon_history.append(cls.agent.epsilon)
        if not done:
            raise "Error"
        cls.weights = np.array(cls.weights)

    def test_epsilon(self):
        self.assertTrue(np.allclose(self.load_data["epsilon"], np.array(self.epsilon_history)))

    def test_states(self):
        self.assertTrue(np.allclose(self.load_data["memory_states"], self.agent.memory.state_memory, rtol=0, atol=1e-2))

    def test_new_states(self):
        self.assertTrue(np.allclose(self.load_data["memory_new_states"], self.agent.memory.new_state_memory, rtol=0, atol=1e-2))

    def test_action(self):
        self.assertTrue(np.allclose(self.load_data["memory_action"], self.agent.memory.action_memory))

    def test_done(self):
        self.assertTrue(np.allclose(self.load_data["memory_dones"], self.agent.memory.terminal_memory))

    def test_rewards(self):
        self.assertTrue(np.allclose(self.load_data["memory_rewards"], self.agent.memory.reward_memory))

    # def test_weights(self):
    #     print(self.load_data["weights"][0][:10])
    #     print(self.weights[0, :10])
    #
    #     self.assertTrue(np.allclose(self.load_data["weights"], self.weights))
