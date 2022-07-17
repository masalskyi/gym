# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from car_racer_env import MCarRacingEnv
from car_racer_model import CarRacerModel
from maslourl.trackers.file_logger import FileLogger
import numpy as np

replay_buffer_size = 1000000
training_batch_size = 64
max_episodes = 300
max_steps = 200
model_backup_frequency_episodes = 25
discount_factor = 0.999
tau = 0.005
noise = 0.1

env = MCarRacingEnv(slide_window_length=4, image_resize=(80, 80))
agent = CarRacerModel(env, replay_buffer_size=10000)
# agent.train(episodes=max_episodes, max_steps_for_episode=max_steps, starting_epsilon=1, epsilon_min=minimum_epsilon,
#             epsilon_decay=epsilon_decay, target_network_replace_frequency_steps=target_network_replace_frequency_steps,
#             training_batch_size=training_batch_size, discount_factor=discount_factor,
#             model_backup_frequency_episodes=model_backup_frequency_episodes, path_to_back_up="./back_ups/",
#             episodes_for_average_tracking=100, file_logger=FileLogger("./logging/log1.csv"))
# agent.test(5, 3000,visualize=True)
print(agent.summary())
t = np.zeros((1, 4, 80, 80))
print(agent.actor.predict(t))
