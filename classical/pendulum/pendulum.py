import tensorflow as tf
import tensorflow.keras as keras
from maslourl.models.maslou_rl import MaslouRLModelDDPGContinuous


# class CriticModel(keras.models.Model):
#     def __init__(self):
#         super(CriticModel, self).__init__()
#
#         self.fc1 = keras.layers.Dense(400, activation="relu")
#         self.fc2 = keras.layers.Dense(300, activation="relu")
#         self.q = keras.layers.Dense(1, activation="linear")
#
#     def call(self, input):
#         state, action = input
#         action_value = self.fc1(tf.concat([state, action], axis=1))
#         action_value = self.fc2(action_value)
#         q = self.q(action_value)
#         return q
#
#
# class ActorModel(keras.models.Model):
#     def __init__(self, n_actions):
#         super(ActorModel, self).__init__()
#         self.n_actions = n_actions
#         self.fc1 = keras.layers.Dense(400, activation="relu")
#         self.fc2 = keras.layers.Dense(300, activation="relu")
#         self.mu = keras.layers.Dense(n_actions, activation="tanh")
#
#     def call(self, state):
#         action = self.fc1(state)
#         action = self.fc2(action)
#         mu = self.mu(action)
#         mu *= 2  # bounds are [-2; 2]
#         return mu
#
# class PendulumDDPGAgent(MaslouRLModelDDPGContinuous):
#     def build_actor_model(self):
#         model = ActorModel(self.n_actions)
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
#         return model
#
#     def build_critic_model(self):
#         model = CriticModel()
#         model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002))
#         return model
class PendulumDDPGAgent(MaslouRLModelDDPGContinuous):
    def build_actor_model(self):
        inputs = keras.layers.Input(shape=self.input_shape)
        x = keras.layers.Dense(400, activation="relu")(inputs)
        x = keras.layers.Dense(300, activation="relu")(x)
        outputs = keras.layers.Dense(self.n_actions, activation="tanh")(x)
        outputs = 2 * outputs
        model = keras.models.Model(inputs, outputs, name="pendulum_actor")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def build_critic_model(self):
        states_input = keras.layers.Input(shape=self.input_shape, name="state_input")
        states_out = keras.layers.Flatten()(states_input)

        actions_input = keras.layers.Input(shape=(self.n_actions,), name="actions_input")
        x = keras.layers.Concatenate()([states_out, actions_input])
        x = keras.layers.Dense(400, activation="relu")(x)
        x = keras.layers.Dense(300, activation="relu")(x)
        output = keras.layers.Dense(1, activation="linear")(x)
        model = keras.models.Model(inputs=[states_input, actions_input], outputs=output, name="pendulum_critic")
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002))
        return model
