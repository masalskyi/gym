from gym.spaces import Box
import gym
import numpy as np
import cv2


def image_preprocess(image, rgb_2_bgr=True, resize=None):
    if rgb_2_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[35:195:2, ::2].reshape(80, 80)

    if resize is not None:
        image = cv2.resize(image, resize)
    return image / 255.0


class PingPongEnv:
    def __init__(self, slide_window_length=3, image_resize=(96, 96), skip_steps = 4):
        self.env = gym.make('PongNoFrameskip-v4')
        self.slide_window_length = slide_window_length
        self.image_resize = image_resize
        self.buffer = np.zeros((slide_window_length, *image_resize))
        self.observation_space = Box(low=0, high=1, shape=self.buffer.shape)
        self.action_space = self.env.action_space
        self.skip_steps = skip_steps

    def reset(self):
        img = self.env.reset()
        img = image_preprocess(img, self.image_resize)
        self.buffer *= 0
        self.buffer[-1] = img
        return self.get_buffer()

    def step(self, action):
        rewards = 0
        for i in range(self.skip_steps):
            new_image, reward, done, info = self.env.step(action)
            rewards += reward
            if done:
                break
        new_image = image_preprocess(new_image, self.image_resize)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = new_image
        return self.get_buffer(), rewards, done, info

    def seed(self, seed_):
        self.env.seed(seed_)

    def get_buffer(self):
        return np.copy(self.buffer)

