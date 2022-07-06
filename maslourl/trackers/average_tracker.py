import numpy as np


class AverageRewardTracker():
    current_index = 0

    def __init__(self, num_rewards_for_average=100):
        self.num_rewards_for_average = num_rewards_for_average
        self.last_x_rewards = []

    def add(self, reward):
        if len(self.last_x_rewards) < self.num_rewards_for_average:
            self.last_x_rewards.append(reward)
        else:
            self.last_x_rewards[self.current_index] = reward
            self.__increment_current_index()

    def __increment_current_index(self):
        self.current_index += 1
        if self.current_index >= self.num_rewards_for_average:
            self.current_index = 0

    def get_average(self):
        return np.average(self.last_x_rewards)
