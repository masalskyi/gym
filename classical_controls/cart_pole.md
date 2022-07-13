# Cart pole v1
[Cart pole](https://www.gymlibrary.ml/environments/classic_control/cart_pole/) is a classical control environment. 
The main purpose of player is balancing the stick. All that player can do is to move right or left. Actions are discrete.
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
