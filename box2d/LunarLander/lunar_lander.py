from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete
from maslourl.models.utils import masked_huber_loss, select_action_epsilon_greedy, get_q_values
import uuid
import shutil
import tensorflow as tf
import numpy as np


class LunarLander2QModel(MaslouRLModel2QDiscrete):
    def __init__(self, env, model, max_step_per_epoch, **params):
        super().__init__(env, model,  **params)
        self.max_step_per_epoch = max_step_per_epoch

    def get_params_for_episode(self):
        return {"fraction_finished": 0}

    def copy_model(self, model):
        backup_file = 'backup_' + str(uuid.uuid4())
        model.save(backup_file)
        new_model = tf.keras.models.load_model(backup_file, custom_objects={'masked_huber_loss': masked_huber_loss()})
        shutil.rmtree(backup_file)
        return new_model

    # returns action that was done, new state, reward, done or not, info
    def step(self, initial_state, target_model, epsilon, params):
        q_values = get_q_values(self.model, initial_state)
        action = select_action_epsilon_greedy(q_values, epsilon)
        new_state, reward, done, info = self.env.step(action)
        params["fraction_finished"] = params["step"] / self.max_step_per_epoch
        new_state = self.state_transform(new_state, params)
        return action, new_state, reward, done, info

    def state_transform(self, state, params):
        return np.append(state, params["fraction_finished"])
