from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete
from maslourl.models.utils import masked_rmse, select_action_epsilon_greedy, get_q_values
import uuid
import shutil
import tensorflow as tf
import numpy as np


class LunarLander2QModel(MaslouRLModel2QDiscrete):
    def get_params_for_episode(self):
        return {"fraction_finished": 0}

    def copy_model(self, model):
        backup_file = 'backup_' + str(uuid.uuid4())
        model.save(backup_file)
        new_model = tf.keras.models.load_model(backup_file, custom_objects={'masked_rmse': masked_rmse()})
        shutil.rmtree(backup_file)
        return new_model

    # returns action that was done, new state, reward, done or not, info
    def step(self, initial_state, target_model, epsilon, params):
        fraction_finished = params["fraction_finished"]
        # state = np.append(initial_state, fraction_finished)
        q_values = get_q_values(self.model, initial_state)
        action = select_action_epsilon_greedy(q_values, epsilon)
        new_state, reward, done, info = self.env.step(action)
        return action, new_state, reward, done, info
