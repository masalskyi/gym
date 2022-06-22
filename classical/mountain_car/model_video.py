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
    inputs = layers.Input(shape=(1, states))
    x = layers.Dense(64, activation="relu") (inputs)
    x = layers.Dense(64, activation="relu") (x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(actions, activation="linear")(x)
    return Model(inputs, outputs, name="mountain_car_player")


model = build_model(2, 3)


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, 3)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
dqn.load_weights("./model/checkpoint_reward_-17.788243832614583.h5f")

env = gym.make('MountainCar-v0')


FRAME_WIDTH = 600
FRAME_HEIGHT = 400
out = cv2.VideoWriter("./results/model_mountain_car.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 60, (FRAME_WIDTH, FRAME_HEIGHT))
episodes = 11
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        prediction = model.predict(np.array(state).reshape((-1,1,2)))
        action = np.argmax(prediction[0])
        # print(prediction)
        state, reward, done, info = env.step(action)
        score += reward
        cv2.putText(img, "After RL:", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Win: {}".format(episode), (FRAME_WIDTH-280, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Lose: 0", (FRAME_WIDTH-150, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        out.write(img)
        # cv2.imshow("game", img)
        # cv2.waitKey(1)
    print("Episode {}: {}".format(episode, score))