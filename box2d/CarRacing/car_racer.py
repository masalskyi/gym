# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from car_racer_env import MCarRacingEnv
from car_racer_model import CarRacerModel
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
max_episodes = 3000
max_steps = 1500
model_backup_frequency_episodes = 25
discount_factor = 0.99
tau = 0.005
noise = [0.1, 0.5]
train_every_step = 4

env = MCarRacingEnv(slide_window_length=2, image_resize=(64, 64))
agent = CarRacerModel(env, replay_buffer_size=replay_buffer_size)
# agent.summary()
agent.train(episodes=max_episodes, max_steps_for_episode=max_steps, train_every_step=train_every_step, noise=noise,
            tau=tau,
            training_batch_size=training_batch_size, discount_factor=discount_factor,
            model_backup_frequency_episodes=model_backup_frequency_episodes, path_to_back_up="./back_ups/",
            episodes_for_average_tracking=50, file_logger=FileLogger("./logging/log1.csv"))
# agent.test(1, 1000, visualize=True)
# t = np.zeros((1,4,80,80))
# print(agent.actor.predict(t))
