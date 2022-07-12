import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete


class LunarLander2QModel(MaslouRLModel2QDiscrete):
    def build_model(self):
        states = 8
        actions = 4
        learning_rate = 0.005
        # regularization_factor = 0.001
        inputs = layers.Input(shape=(states,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(actions, activation="linear")(x)
        model = Model(inputs, outputs, name="LunarLander")
        model.compile(Adam(learning_rate=learning_rate), loss="mse")
        return model