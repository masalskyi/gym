from gym.spaces import Box
import gym
import numpy as np
import cv2


def image_preprocess(image, rgb_2_bgr=True, resize=None):
    print(image.shape)
    if rgb_2_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        image = cv2.resize(image, resize)
    return image / 255.0

class MCarRacingEnv:
    def __init__(self, slide_window_length=3, image_resize=(96, 96)):
        self.env = gym.make("CarRacing-v0")
        self.slide_window_length = slide_window_length
        self.image_resize = image_resize
        self.buffer = np.zeros((slide_window_length, *image_resize))
        self.observation_space = Box(low=0, high=1, shape=self.buffer.shape)
        self.action_space = self.env.action_space

    def reset(self):
        img = self.env.reset()
        print("Before reset", img.shape)
        img = image_preprocess(img, resize=self.image_resize)
        del self.buffer
        self.buffer = np.zeros((self.slide_window_length, *self.image_resize))
        self.buffer[-1] = img
        return self.buffer

    def step(self, action):
        new_image, reward, done, info = self.env.step(action)
        print("After step", new_image.shape)
        new_image = image_preprocess(new_image, resize=self.image_resize)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = new_image
        return self.buffer

    def render(self, mode="human"):
        self.env.render(mode)