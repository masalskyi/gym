import gym
from pendulum import PendulumDDPGAgent
from maslourl.trackers.file_logger import FileLogger

env = gym.make('Pendulum-v1', g=9.81)

replay_buffer_size = 1000000
training_batch_size = 64
max_episodes = 300
max_steps = 200
model_backup_frequency_episodes = 10
discount_factor = 0.99
tau = 0.005
noise = 0.1

agent = PendulumDDPGAgent(env)

agent.train(episodes=max_episodes, max_steps_for_episode=max_steps, tau=0.005, noise=noise,
            training_batch_size=training_batch_size,
            discount_factor=discount_factor, model_backup_frequency_episodes=model_backup_frequency_episodes,
            path_to_back_up="./back_ups/",
            episodes_for_average_tracking=20, file_logger=FileLogger("./logging/log2.csv"))
