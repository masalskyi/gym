# Opean AI Gym
## Introduction
This projects aims to improve my skills in RL. All Environments that i would use can fe found [here](https://www.gymlibrary.ml/)
All algorithms are created in python + tensorflow + keras + keras-rl
## Classical controls

### Cart pole v1
[Cart pole](https://www.gymlibrary.ml/environments/classic_control/cart_pole/) is a classical control environment. 
The main purpose of player is balancing the stick. All that player can do is to move right or left.
The game ends when:
<ul>
    <li>Pole Angle is greater than ±12°</li>
    <li>Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)</li>
    <li>Episode length is greater than 500</li>
</ul>
<p align="center">
    <iframe width="560" height="315" src="https://www.youtube.com/embed/-0jJZjB42ZU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>
The solution is based on deep q-learning algorithm. The model has the structure:
```python
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
def build_model(states, actions):
    model = Sequential()
    model.add(layers.Flatten(input_shape=(1,states)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(actions, activation="linear"))
    return model
```
The model was trained on 100000 steps with Adam optimizer(lr=1e-3) and BoltzmannQPolicy, also using checkpoint callback
to save the model that achieve the best rewards. Callbacks were taken from [here](https://github.com/guillemhub/DRLDBackEnd/blob/master/callbacks.py#L179).

### Mountain car v0
[Mountain Car](https://www.gymlibrary.ml/environments/classic_control/mountain_car/) is a classical control environment. 
The main purpose of player is to achieve the flag. All that player can do is to apply force to 
right or to left or not to move at all.
The game ends when:
<ul>
 <li>The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)</li>
 <li>The length of the episode is 200.</li>
</ul>
<iframe width="560" height="315" src="https://www.youtube.com/embed/9ZyPZ-KfGF8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

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