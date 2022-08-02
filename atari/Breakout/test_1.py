import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Orthogonal
from maslourl.models.ppo_tf import MaslouRLModelPPODiscrete
import gym
import numpy as np
class BreakoutAgent(MaslouRLModelPPODiscrete):
    def __init__(self):
        self.feature_size = 512
        super(BreakoutAgent, self).__init__()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make("BreakoutNoFrameskip-v4")
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    def build_feature_extractor(self) -> keras.Model:
        input = layers.Input(shape=(4, 84, 84))
        x = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(input)
        x = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(x)
        output = layers.Dense(units=self.feature_size, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2))) (x)
        model = keras.Model(inputs=input, outputs=output, name="FeatureExtractor")
        return model

    def build_actor(self) -> keras.Model:
        input = layers.Input(shape=(self.feature_size,))
        output = layers.Dense(self.n_actions, activation="softmax", kernel_initializer=Orthogonal(0.01))(input)
        model = keras.Model(inputs=input, outputs=output, name="Actor")
        return model

    def build_critic(self) -> keras.Model:
        input = layers.Input(shape=(self.feature_size,))
        output = layers.Dense(1, activation="linear", kernel_initializer=Orthogonal(1))(input)
        model = keras.Model(inputs=input, outputs=output, name="Critic")
        return model

agent = BreakoutAgent()
