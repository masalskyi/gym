# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from ping_pong_env import PingPongEnv
from ping_pong_model import PingPongModel
from maslourl.trackers.file_logger import FileLogger
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

replay_buffer_size = 10000
training_batch_size = 64
max_episodes = 1000
max_steps = 3000
target_network_replace_frequency_steps = 250
model_backup_frequency_episodes = 25
starting_epsilon = 1
minimum_epsilon = 0.01
epsilon_decay = 0.9998
discount_factor = 0.99

env = PingPongEnv(slide_window_length=4, image_resize=(80, 80), skip_steps=4)
agent = PingPongModel(env, replay_buffer_size=replay_buffer_size)
agent.summary()
agent.train(episodes=max_episodes, max_steps_for_episode=max_steps, starting_epsilon=starting_epsilon, epsilon_decay=epsilon_decay, epsilon_min=minimum_epsilon,
            training_batch_size=training_batch_size, discount_factor=discount_factor,
            model_backup_frequency_episodes=model_backup_frequency_episodes, path_to_back_up="./back_ups/",
            episodes_for_average_tracking=50, file_logger=FileLogger("./logging/log1.csv"))
# agent.test(1, 1000, visualize=True)
# t = np.zeros((1,4,80,80))
# print(agent.actor.predict(t))
