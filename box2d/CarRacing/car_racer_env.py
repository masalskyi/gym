from gym.spaces import Box
import gym
import numpy as np
import cv2


def image_preprocess(image, rgb_2_bgr=True, resize=None):
    image = image[:-50]

    image[np.where((np.logical_and(image >= [101, 203, 101], image <= [101, 230, 101])).all(axis=2))] = np.array(
        [101, 203, 101])
    image[np.where((np.logical_and(image >= [101, 101, 101], image <= [106, 106, 106])).all(axis=2))] = np.array(
        [106, 106, 106])
    if rgb_2_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image[278:321, 285:315] = 0
    if resize is not None:
        image = cv2.resize(image, resize)
    image = cv2.Canny(image, 100, 200)

    return image / 255.0


class MCarRacingEnv:
    def __init__(self, slide_window_length=3, image_resize=(96, 96)):
        self.env = gym.make("CarRacing-v2")
        self.slide_window_length = slide_window_length
        self.image_resize = image_resize
        self.buffer = np.zeros((slide_window_length, *image_resize))
        self.observation_space = Box(low=0, high=1, shape=self.buffer.shape)
        self.action_space = Box(low=-1, high=1, shape=(2,))

    def reset(self):
        img = self.env.reset()
        img = image_preprocess(img, resize=self.image_resize)
        self.buffer *= 0
        self.buffer[-1] = img
        return self.buffer

    def step(self, action):
        action = self.process_action(action)
        new_image, reward, done, info = self.env.step(action)
        new_image = image_preprocess(new_image, resize=self.image_resize)
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = new_image
        return self.buffer, reward, done, info

    def render(self, mode="human"):
        img = self.env.render(mode="rgb_array")
        cv2.imshow("Game", img)
        img = image_preprocess(img, resize=np.array(self.image_resize))
        cv2.imshow("Processed", img)
        cv2.waitKey(30)

    def process_action(self, action):
        return np.array([action[0], np.clip(action[1], 0, 1), -np.clip(action[1], -1, 0)])
