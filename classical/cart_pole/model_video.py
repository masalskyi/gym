from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import gym
import cv2
import numpy as np

def build_model(states, actions):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(1, states)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model


model = build_model(4, 2)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=actions, nb_steps_warmup=100,
                   target_model_update=1e-2)
    return dqn


dqn = build_agent(model, 2)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
dqn.load_weights("./model/model.h5f")

env = gym.make("CartPole-v1")


FRAME_WIDTH = 600
FRAME_HEIGHT = 400
out = cv2.VideoWriter("./results/model_cart_pole.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 120, (FRAME_WIDTH, FRAME_HEIGHT))
episodes = 5
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        prediction = model.predict(np.array(state).reshape((-1,1,4)))
        action = np.argmax(prediction[0])
        # print(prediction)
        state, reward, done, info = env.step(action)
        score += reward
        cv2.putText(img, "After RL(x6 speed):", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Win: {}".format(episode), (FRAME_WIDTH-280, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Lose: 0", (FRAME_WIDTH-150, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Score: {}".format(score), (FRAME_WIDTH - 280, 60), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        out.write(img)
        # cv2.imshow("game", img)
        # cv2.waitKey(1)
    print("Episode {}: {}".format(episode, score))