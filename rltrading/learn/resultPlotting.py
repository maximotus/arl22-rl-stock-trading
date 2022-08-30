import matplotlib.pyplot as plt
import numpy as np


def plotResults(result_memory):
    print(result_memory)
    i = 0
    j = 0
    k = 0
    for e in (result_memory):
        if e.action == 1:
            i += 1
        if e.action == 2:
            j += 1
        if e.action == 0:
            k += 1
    print("Action 0: ", k)
    print("Action 1: ", i)
    print("Acton 2:", j)
    time = range(len(result_memory))
    fig, axes = plt.subplots(2, 1)
    res = list(map(lambda x: x.reward, result_memory))
    cum_rewards = np.cumsum(res)
    ax = axes[0]
    ax.plot(time, cum_rewards)
    ax.set(
        xlabel="time", ylabel="cumulative rewards", title="cumulative rewards over time"
    )
    ax.grid()

    ax2 = axes[1]
    bin = np.arange(4)
    ax2.hist(list(map(lambda x: x.action, result_memory)), bins=bin, edgecolor="black")
    ax2.set_xticks(bin + 0.5, bin)
    ax2.set(
        xlabel="action",
        ylabel="n times chosen",
        title="the amount each action got chosen",
    )

    plt.show()
