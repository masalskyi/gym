import gym
import cv2
import numpy as np
from acrobot import Acrobot2QModel




env = gym.make('Acrobot-v1')
agent = Acrobot2QModel(env)
agent.load_model("model/model.h5")


FRAME_WIDTH = 500
FRAME_HEIGHT = 500
out = cv2.VideoWriter("./results/model_video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (FRAME_WIDTH, FRAME_HEIGHT))
episodes = 10
for episode in range(episodes):
    env.seed(episode)
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        action = agent.choose_action(state)
        # print(prediction)
        state, reward, done, info = env.step(action)
        score += reward
        cv2.putText(img, "After RL:", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        delta = 1 if done else 0
        cv2.putText(img, "Win: {}".format(episode+delta), (FRAME_WIDTH-280, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Lose: 0", (FRAME_WIDTH-150, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Score: {}".format(score), (FRAME_WIDTH - 280, 60), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        out.write(img)
        if done:
            img = env.render(mode="rgb_array")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img, "After RL:", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
            delta = 1 if done else 0
            cv2.putText(img, "Win: {}".format(episode + delta), (FRAME_WIDTH - 280, 30), cv2.FONT_ITALIC, 0.8,
                        (0, 0, 0), thickness=2)
            cv2.putText(img, "Lose: 0", (FRAME_WIDTH - 150, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
            cv2.putText(img, "Score: {}".format(score), (FRAME_WIDTH - 280, 60), cv2.FONT_ITALIC, 0.8, (0, 0, 0),
                        thickness=2)
            for i in range(25):
                out.write(img)
    print("Episode {}: {}".format(episode, score))