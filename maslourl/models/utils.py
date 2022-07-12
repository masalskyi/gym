import numpy as np
import random


def get_q_values(model, state):
    input = state[np.newaxis, ...]
    return model.predict(input, verbose=0)[0]


def get_multiple_q_values(model, states):
    return model.predict(states, verbose=0)


def select_action_epsilon_greedy(q_values, epsilon):
    random_value = random.uniform(0, 1)
    if random_value < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)


def select_best_action(q_values):
    return np.argmax(q_values)

