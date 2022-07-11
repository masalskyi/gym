import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete
from maslourl.models.utils import masked_huber_loss, select_action_epsilon_greedy, get_q_values

import uuid
import shutil
import tensorflow as tf
import numpy as np


class LunarLander2QModel(MaslouRLModel2QDiscrete):
    def __init__(self, env, max_step_per_epoch, **params):
        super().__init__(env, **params)
        self.max_step_per_epoch = max_step_per_epoch


    def copy_model(self, model):
        backup_file = 'backup_' + str(uuid.uuid4())
        model.save(backup_file)
        new_model = tf.keras.models.load_model(backup_file, custom_objects={'masked_huber_loss': masked_huber_loss()})
        shutil.rmtree(backup_file)
        return new_model

    # returns action that was done, new state, reward, done or not, info
    def step(self, initial_state, epsilon):
        q_values = get_q_values(self.model, initial_state)
        action = select_action_epsilon_greedy(q_values, epsilon)
        new_state, reward, done, info = self.env.step(action)
        return action, new_state, reward, done, info


    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path, custom_objects={'masked_huber_loss': masked_huber_loss()})

    def build_model(self):
        states = 8
        actions = 4
        learning_rate = 0.001
        regularization_factor = 0.001
        inputs = layers.Input(shape=(states,))
        x = layers.Dense(32, activation="relu", kernel_regularizer=l2(regularization_factor))(inputs)
        x = layers.Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor))(x)
        outputs = layers.Dense(actions, activation="linear", kernel_regularizer=l2(regularization_factor))(x)
        model = Model(inputs, outputs, name="LunarLander")
        model.compile(Adam(learning_rate=learning_rate), loss=masked_huber_loss())
        return model