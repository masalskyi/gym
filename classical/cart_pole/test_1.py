import argparse
import os
import time

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.optimizers import Adam
from ppo_tf import MaslouRLModelPPODiscrete
import gym
from distutils.util import strtobool
import numpy as np
# class EpisodicLifeEnv(gym.Wrapper):
#     """
#     Make end-of-life == end-of-episode, but only reset on true game over.
#     Done by DeepMind for the DQN and co. since it helps value estimation.
#
#     :param env: the environment to wrap
#     """
#
#     def __init__(self, env: gym.Env):
#         gym.Wrapper.__init__(self, env)
#         self.lives = 0
#         self.was_real_done = True
#
#     def step(self, action: int) :
#         obs, reward, done, info = self.env.step(action)
#         self.was_real_done = done
#         # check current lives, make loss of life terminal,
#         # then update lives to handle bonus lives
#         lives = self.env.unwrapped.ale.lives()
#         if 0 < lives < self.lives:
#             # for Qbert sometimes we stay in lives == 0 condtion for a few frames
#             # so its important to keep lives > 0, so that we only reset once
#             # the environment advertises done.
#             done = True
#         self.lives = lives
#         return obs, reward, done, info
#
#
#     def reset(self, **kwargs) -> np.ndarray:
#         """
#         Calls the Gym environment reset, only when lives are exhausted.
#         This way all states are still reachable even though lives are episodic,
#         and the learner need not know about any of this behind-the-scenes.
#
#         :param kwargs: Extra keywords passed to env.reset() call
#         :return: the first observation of the environment
#         """
#         if self.was_real_done:
#             obs = self.env.reset(**kwargs)
#         else:
#             # no-op step to advance from terminal/lost life state
#             obs, _, _, _ = self.env.step(0)
#         self.lives = self.env.unwrapped.ale.lives()
#         return obs
class BreakoutAgent(MaslouRLModelPPODiscrete):
    def __init__(self):
        super(BreakoutAgent, self).__init__()

    def make_env(self, seed, idx, capture_video, capture_every_n_episode, run_name):
        def thunk():
            env = gym.make("CartPole-v1")
            # env = EpisodicLifeEnv(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = gym.wrappers.ResizeObservation(env, (84, 84))
            # env = gym.wrappers.GrayScaleObservation(env)
            # env = gym.wrappers.FrameStack(env, 4)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    def build_model(self) -> keras.Model:
        # input = layers.Input(shape=(4, 84, 84))
        # x = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(input)
        # x = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(x)
        # x = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)), data_format='channels_first')(x)
        # x = layers.Flatten()(x)
        # hidden = layers.Dense(units=self.feature_size, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)))(x)
        # actions = layers.Dense(self.n_actions, activation="softmax", kernel_initializer=Orthogonal(0.01))(hidden)
        # value = layers.Dense(1, activation="linear", kernel_initializer=Orthogonal(1))(hidden)
        # model = keras.Model(inputs=input, outputs=[actions, value], name="BreakoutModel")
        # return model

        input = layers.Input(shape=(4,))
        x = layers.Dense(128, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)))(input)
        hidden = layers.Dense(units=64, activation="relu", kernel_initializer=Orthogonal(np.sqrt(2)))(x)
        actions = layers.Dense(self.n_actions, activation="softmax", kernel_initializer=Orthogonal(0.01))(hidden)
        value = layers.Dense(1, activation="linear", kernel_initializer=Orthogonal(1))(hidden)
        model = keras.Model(inputs=input, outputs=[actions, value], name="Cartpole")
        return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment")
    parser.add_argument('--gym-id', type=str, default="CartPole-v1",
                        help='the id of the openai gym environment')
    parser.add_argument('--average-reward-2-save', type=int, default=20,
                        help="Tracking the average reward with specified length and save the best model")
    parser.add_argument('--save-best-to-wandb', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to save the best model to w&b")
    parser.add_argument('--learning_rate', type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument('--verbose', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle whether to verbose on console")
    parser.add_argument('--seed', type=int, default=1, help="seed of the experiment")
    parser.add_argument('--total-timesteps', type=int, default=10000000, help="total timesteps of the experiment")
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled `torch.backend.cudnn.deterministic=False")
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will not be enabled by default")
    parser.add_argument('--track', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with w&b")
    parser.add_argument('--wandb-project-name', type=str, default="rl-ppo-cart-pole", help="the w&b project name")
    parser.add_argument('--wandb-entity', type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent perfomance")
    parser.add_argument('--capture-every-n-video', type=int, default=50, help="capture every nth video")
    parser.add_argument('--num-envs', type=int, default=8,
                        help="the number of parallel game environments")
    parser.add_argument('--num-steps', type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument('--anneal-lr', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="use General Andvantage estimation for advantage computation")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help="the lambda for GAE")
    parser.add_argument('--num-minibutches', type=int, default=4,
                        help="the number of minibatches")
    parser.add_argument('--update-epochs', type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument('--norm_adv', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coeficient")
    parser.add_argument('--clip-vloss', type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="toggles whether or not to use a cliped loss for the value function as per the paper")
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient for entropy loss")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient for loss of value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help="the maximum norm of gradient clipping")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibutches)
    return args


if __name__ == '__main__':
    args = parse_args()
    agent = BreakoutAgent()
    agent.summary()
    run_name = f"{args.gym_id}_{args.exp_name}_{args.seed}_{int(time.time())}"
    agent.train(learning_rate=args.learning_rate, num_steps=args.num_steps,
                num_envs=args.num_envs, seed=args.seed,
                capture_video=args.capture_video, capture_every_n_video=args.capture_every_n_video, run_name=run_name,
                total_timesteps=args.total_timesteps, anneal_lr=args.anneal_lr, gae=args.gae, discount_gamma=args.gamma,
                gae_lambda=args.gae_lambda, update_epochs=args.update_epochs,
                minibatches=args.num_minibutches, norm_adv=args.norm_adv, clip_coef=args.clip_coef,
                clip_vloss=args.clip_vloss, ent_coef=args.ent_coef, vf_coef=args.vf_coef, track=args.track,
                wandb_project_name=args.wandb_project_name, wandb_entity=args.wandb_entity, config=args)