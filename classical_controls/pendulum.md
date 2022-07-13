# Pendulum-v1
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