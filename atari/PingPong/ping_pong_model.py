import tensorflow.keras.backend as K

from maslourl.models.maslou_rl import MaslouRLModel2QDiscrete
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers

class PingPongModel(MaslouRLModel2QDiscrete):
    def build_model(self):
        state_input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=32, kernel_size=8, strides=2, activation="relu", data_format='channels_first')(state_input)
        x = layers.Conv2D(filters=64, kernel_size=4, strides=4, activation="relu", data_format='channels_first')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", data_format='channels_first')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        v = layers.Dense(1, activation="linear")(x)
        a = layers.Dense(self.n_actions, activation="linear")(x)
        outputs = v + a - K.mean(a)
        model = Model(inputs=state_input, outputs=outputs, name="PingPong")
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
        return model

