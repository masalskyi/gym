from car_racer_env import MCarRacingEnv
from car_racer_model import CarRacerModel
from maslourl.trackers.file_logger import FileLogger
import numpy as np
import tensorflow as tf
import cv2

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

FRAME_WIDTH = 600
FRAME_HEIGHT = 400
out = cv2.VideoWriter("./results/model_video_rotating.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 10, (FRAME_WIDTH, FRAME_HEIGHT))
episodes = 1
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        action = agent.choose_action(state)[0]
        # print(prediction)
        state, reward, done, info = env.step(action)
        score += reward
        out.write(img)
out.release()