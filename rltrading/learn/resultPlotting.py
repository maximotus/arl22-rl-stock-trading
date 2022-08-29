import matplotlib.pyplot as plt
import numpy as np

def plotResults(result_memory):
	time = range(len(result_memory))
	fig, axes =  plt.subplots(2, 1)
	res = list(map(lambda x: x.reward, result_memory))
	cum_rewards = np.cumsum(res)
	ax = axes[0]
	ax.plot(time, cum_rewards)
	ax.set(xlabel='time', ylabel='cumulative rewards', title='cumulative rewards over time')
	ax.grid()

	ax2 = axes[1]
	bin = np.arange(3)
	ax2.hist(list(map(lambda x: x.reward, result_memory)), bins=bin, edgecolor='black')
	ax2.set_xticks(bin+.5, bin)
	ax2.set(xlabel='action', ylabel='n times chosen', title='the amount each action got chosen')

	plt.show()