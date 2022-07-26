import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("./logging/log1.csv", sep=';')
plt.figure(figsize=(10, 6))
plt.plot([800] * len(data['average']), c="gray", label="Good performance")
plt.plot(data['reward'], c="orange", label="Reward")
plt.plot(data['average'], c="blue", label="Average reward")

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='upper right')
plt.show()