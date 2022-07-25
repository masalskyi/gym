import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("./logging/log1.csv", sep=';')
plt.figure(figsize=(10, 6))
plt.plot([800] * len(data['average']), c="gray")
plt.plot(data['average'], c="blue")
plt.plot(data['reward'], c="orange")
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(['Good perfomance', 'Average reward', 'Reward'], loc='upper right')
plt.title('Reward')
plt.show()