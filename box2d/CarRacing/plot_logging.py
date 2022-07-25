import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("./logging/log1.csv", sep=';')
plt.figure(figsize=(20, 12))

plt.plot(data['average'])
plt.plot(data['reward'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(['Average reward', 'Reward'], loc='upper right')
plt.title('Reward')
plt.show()