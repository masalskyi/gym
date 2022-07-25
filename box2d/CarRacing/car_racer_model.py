import tensorflow.keras.backend

from maslourl.models.maslou_rl import MaslouRLModelDDPGContinuous
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers


# class CarRacerModel(MaslouRLModelDDPGContinuous):
#     def build_critic_model(self):
#         state_input = layers.Input(shape=self.input_shape)
#         x = layers.Conv2D(filters=32, kernel_size=8, activation="relu", data_format='channels_first')(state_input)
#         x = layers.Conv2D(filters=32, kernel_size=8, activation="relu", data_format='channels_first')(x)
#         x = layers.MaxPool2D(pool_size=2)(x)
#         x = layers.Conv2D(filters=64, kernel_size=4, activation="relu", data_format='channels_first')(x)
#         x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", data_format='channels_first')(x)
#         x = layers.Flatten()(x)
#
#         action_input = layers.Input(shape=(self.n_actions,))
#         x = layers.Concatenate()([x, action_input])
#         x = layers.Dense(100, activation="relu")(x)
#
#         outputs = layers.Dense(1, activation="linear")(x)
#         model = Model(inputs=[state_input, action_input], outputs=outputs, name="CarRacer_critic")
#         model.compile(optimizer=Adam(learning_rate=0.002), loss="mse")
#         return model
#
#     def build_actor_model(self):
#         state_input = layers.Input(shape=self.input_shape)
#         x = layers.Conv2D(filters=32, kernel_size=8, activation="relu", data_format='channels_first')(state_input)
#         x = layers.Conv2D(filters=32, kernel_size=8, activation="relu", data_format='channels_first')(x)
#         x = layers.MaxPool2D(pool_size=2)(x)
#         x = layers.Conv2D(filters=64, kernel_size=4, activation="relu", data_format='channels_first')(x)
#         x = layers.Conv2D(filters=128, kernel_size=3, activation="relu", data_format='channels_first')(x)
#         x = layers.Flatten()(x)
#         x = layers.Dense(100, activation="relu")(x)
#
#         rotation = layers.Dense(1, activation="tanh")(x)
#         gas = layers.Dense(1, activation="sigmoid")(x)
#         breaking = layers.Dense(1, activation="sigmoid")(x)
#         output = layers.Concatenate()([rotation, gas, breaking])
#         model = Model(inputs=state_input, outputs=output, name="CarRacer_actor")
#         model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
#         return model

class CarRacerModel(MaslouRLModelDDPGContinuous):
    def build_critic_model(self):
        state_input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation="relu", data_format='channels_first')(state_input)
        x = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation="relu", data_format='channels_first')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation="relu", data_format='channels_first')(x)
        x = layers.Flatten()(x)

        action_input = layers.Input(shape=(self.n_actions,))
        x = layers.Concatenate()([x, action_input])
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        outputs = layers.Dense(1, activation="linear")(x)
        model = Model(inputs=[state_input, action_input], outputs=outputs, name="CarRacer_critic")
        model.compile(optimizer=Adam(learning_rate=0.0002), loss="mse")
        return model

    def build_actor_model(self):
        state_input = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation="relu", data_format='channels_first')(state_input)
        x = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation="relu", data_format='channels_first')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation="relu", data_format='channels_first')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        output = layers.Dense(self.n_actions, activation="tanh")(x)
        model = Model(inputs=state_input, outputs=output, name="CarRacer_actor")
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="mse")
        return model
