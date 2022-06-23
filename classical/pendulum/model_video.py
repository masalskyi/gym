from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import tensorflow.keras.layers as layers
from rl.random import OrnsteinUhlenbeckProcess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import gym
import cv2
import numpy as np

def build_agent_model(states, actions):
    inputs = layers.Input(shape=(1, states))
    x = layers.Dense(64, activation="relu") (inputs)
    x = layers.Dense(64, activation="relu") (x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(actions, activation="tanh")(x)
    outputs = 2 * outputs
    return Model(inputs, outputs, name="pendulum_agent")

def build_critic_model(states, actions):
    states_input = layers.Input(shape=(1, states), name="state_input")
    states_out = layers.Dense(64, activation="relu")(states_input)
    states_out = layers.Flatten()(states_out)

    actions_input = layers.Input(shape=(actions), name="actions_input")
    actions_out = layers.Dense(64, activation="relu")(actions_input)

    x = layers.Concatenate()([states_out, actions_out])
    x = layers.Dense(128,activation="relu")(x)
    x = layers.Dense(128,activation="relu")(x)
    output = layers.Dense(1, activation="linear")(x)
    return Model(inputs=[states_input, actions_input], outputs=output, name="pendulum_critic"), actions_input

agent = build_agent_model(3, 1)
critic, actions_input = build_critic_model(3, 1)


def build_agent(actions):
    memory = SequentialMemory(limit=50000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=actions, theta=.15, mu=0., sigma=.3)
    dqn = DDPGAgent(actions, agent, critic, actions_input, memory, random_process= random_process)
    return dqn


dqn = build_agent(1)
dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])

dqn.load_weights("./model/model.h5f")

env = gym.make('Pendulum-v1', g=9.81)
FRAME_WIDTH = 500
FRAME_HEIGHT = 500

out = cv2.VideoWriter("./results/model_video.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 60, (FRAME_WIDTH, FRAME_HEIGHT))
episodes = 17
c = 0
for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        img = env.render(mode="rgb_array")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        prediction = agent.predict(np.array(state).reshape((-1,1,3)))
        new_state, reward, done, info = env.step(prediction[0])
        if np.linalg.norm(state - new_state) < 1e-6:
            c+=1
            if c > 10:
                done = 1
        else:
            c = 0
        state = new_state
        score += reward
        cv2.putText(img, "After RL:", (10, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Win: {}".format(episode), (FRAME_WIDTH-280, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        cv2.putText(img, "Lose: 0", (FRAME_WIDTH-150, 30), cv2.FONT_ITALIC, 0.8, (0, 0, 0), thickness=2)
        out.write(img)
        # cv2.imshow("game", img)
        # cv2.waitKey(1)
    print("Episode {}: {}".format(episode, score))