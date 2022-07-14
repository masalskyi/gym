import numpy as np
import cv2
import gym
import random

env = gym.make('Acrobot-v1')
FRAME_WIDTH = 500
FRAME_HEIGHT = 500
out = cv2.VideoWriter("./results/random_video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (FRAME_WIDTH, FRAME_HEIGHT))

episodes = 5
for episode in range(episodes+1):
    env.seed(episode)
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        action = random.choice([0, 1, 2])
        n_state, reward, done, info = env.step(action)
        score += reward
        cv2.putText(img, "Before RL:", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Win: 0", (FRAME_WIDTH-280, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        delta = 1 if done else 0
        cv2.putText(img, "Lose: {}".format(episode+delta), (FRAME_WIDTH-160, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Score: {}".format(score), (FRAME_WIDTH-280, 60), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)

        out.write(img)
        # cv2.imshow("game", img)
        # cv2.waitKey(1)
    print("Episode {}: {}".format(episode, score))

cv2.destroyAllWindows()
out.release()
