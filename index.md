# Opean AI Gym
## Introduction
This projects aims to improve my skills in RL. All Environments that i would use can be found [here](https://www.gymlibrary.ml/)
All algorithms are created in python + tensorflow + keras + keras-rl
## Classical controls
<ul>
    [Cart pole](classical_controls/cart_pole.md)
</ul>

### Mountain car v0
[Mountain Car](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) is a classical control environment. 
The main purpose of player is to achieve the flag. All that player can do is to apply force to 
right or to left or not to move at all.
The game ends when:
<ul>
 <li>The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)</li>
 <li>The length of the episode is 200.</li>
</ul>
<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/9ZyPZ-KfGF8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
The solution is based on deep q-learning algorithm. The model has the structure:
```python
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
def build_model(states, actions):
    inputs = layers.Input(shape=(1, states))
    x = layers.Dense(64, activation="relu") (inputs)
    x = layers.Dense(64, activation="relu") (x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(actions, activation="linear")(x)
    return Model(inputs, outputs, name="mountain_car_player")
```
The model was trained on 100000 steps with Adam optimizer(lr=1e-3) and BoltzmannQPolicy, also using checkpoint callback
to save the model that achieve the best rewards. The main problem of the game is a reward function. It gives -1 for 
every step that player not achieved a goal. But it does not provide any usefull information of how to do this. Using a theoretical
materials from [paper](https://arxiv.org/pdf/2011.02669.pdf), and [medium post](https://medium.com/@BonsaiAI/deep-reinforcement-learning-models-tips-tricks-for-writing-reward-functions-a84fe525e8e0)
I performed a reward shaping in a such way:

```python
from gym.envs.classic_control.mountain_car import MountainCarEnv

class MountainCarModifiedReward(MountainCarEnv):
    def step(self, action: int):
        previous_state = self.state
        new_state, reward, done, info = super().step(action)
        modified_reward = reward + 300 * (0.95 * abs(new_state[1]) - abs(previous_state[1]))
        if new_state[0] >= 0.5:
            modified_reward += 100
        return new_state, modified_reward, done, info
```

The main idea was that to climb a hill the car need a big velocity, so i used a potential reward shaping 
to stimulate a high velocity. I also add +100 reward for actual winning a game to stimulate to actually win 
a game but not to achieve a big score using only velocity rewards.

### Pendulum-v1
The inverted [pendulum](https://www.gymlibrary.ml/environments/classic_control/pendulum/) swingup problem 
is based on the classic problem in control theory. 
The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an 
upright position, with its center of gravity right above the fixed point.
The episode terminates at 200 time steps.
<p align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/1kgrQNWCbEE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
This is a type of RL problems where actions are continuous variables, 
so the models I used before are not suitable here. The solution is based on deep q-learning algorithm using 
[DDPGAgent](https://keras-rl.readthedocs.io/en/latest/agents/ddpg/). Algorithm needs two models. 
Agent (actor) model, and critic model. Agent model by observation predict an action. Critic model by 
observation and action predict the future rewards. Agent and critic models have the following architecture:

```python
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model

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
```
The model was trained on 10000 steps with Adam optimizer(lr=1e-3) and random process 
as OrnsteinUhlenbeckProcess(size=actions, theta=.15, mu=0., sigma=.3)