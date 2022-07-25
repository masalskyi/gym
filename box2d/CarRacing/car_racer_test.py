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

env = MCarRacingEnv(slide_window_length=2, image_resize=(80, 80), skip_steps=3)
agent = CarRacerModel(env, train_state=False, replay_buffer_size=0)
agent.load_model("./back_ups/model_750.h5")

agent.test(1, 1000, visualize=True)
# t = np.zeros((1,4,80,80))
# print(agent.actor.predict(t))
