import os
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete


class Acrobot2QModel(MaslouRLModel2QDiscrete):
    def build_model(self):
        states = self.env.observation_space.shape[0]
        actions = self.env.action_space.n
        learning_rate = 0.0005
        # regularization_factor = 0.001
        inputs = layers.Input(shape=(states,))
        x = layers.Dense(64, activation="relu")(inputs)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(actions, activation="linear")(x)
        model = Model(inputs, outputs, name="LunarLander")
        model.compile(Adam(learning_rate=learning_rate), loss="mse")
        return model