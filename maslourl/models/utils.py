import numpy as np
import random
import tensorflow as tf
import keras.backend as K


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


def calculate_target_values(model, target_model, state_transitions, discount_factor, outputs):
    states = []
    new_states = []
    for transition in state_transitions:
        states.append(transition.old_state)
        new_states.append(transition.new_state)

    new_states = np.array(new_states)

    q_values_new_state = get_multiple_q_values(model, new_states)
    q_values_new_state_target_model = get_multiple_q_values(target_model, new_states)

    targets = []
    for index, state_transition in enumerate(state_transitions):
        best_action = select_best_action(q_values_new_state[index])
        best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

        target_vector = [0] * outputs
        target_vector[state_transition.action] = target_value
        targets.append(target_vector)

    return np.array(targets)


def masked_rmse(mask_value=0):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        return K.sqrt(K.sum(masked_squared_error) / K.sum(mask_true))

    f.__name__ = 'masked_rmse'
    return f


def masked_huber_loss(mask_value=0, clip_delta=1):
    def f(y_true, y_pred):
        error = y_true - y_pred
        cond = K.abs(error) < clip_delta
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
        linear_loss = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
        huber_loss = tf.where(cond, masked_squared_error, linear_loss)
        return K.sum(huber_loss) / K.sum(mask_true)

    f.__name__ = 'masked_huber_loss'
    return f
