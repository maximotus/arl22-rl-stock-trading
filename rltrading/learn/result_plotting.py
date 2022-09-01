import matplotlib.pyplot as plt
import numpy as np
from rltrading.gym.environment import Positions

def plot_results(result_memory):
    # print(result_memory)
    i = 0
    j = 0
    k = 0
    for e in result_memory:
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
    fig, axes = plt.subplots(2, 2, figsize=(20,8))
    res = list(map(lambda x: x.reward, result_memory))
    cum_rewards = np.cumsum(res)
    ax = axes[0][0]
    ax.plot(time, cum_rewards)
    ax.set(
        xlabel="time", ylabel="cumulative rewards", title="cumulative rewards over time"
    )
    ax.grid()

    ax2 = axes[1][0]
    bin = np.arange(4)
    ax2.hist(list(map(lambda x: x.action, result_memory)), bins=bin, edgecolor="black")
    ax2.set_xticks(bin + 0.5, bin)
    ax2.set(
        xlabel="action",
        ylabel="n times chosen",
        title="the amount each action got chosen",
    )

    ax3 = axes[0][1]
    prices = list(map(lambda x: x.observation[len(x.observation)-1][0], result_memory))
    ax3.plot(time, prices)
    actions = list(map(lambda x: x.action,  result_memory))
    action = actions[0]
    # since we start with a long first datapoint 0 is a long
    longs = [0]
    long_prices = [prices[0]]
    shorts = []
    short_prices = []
    for i in range(1, len(actions)):
        if(actions[i] != action):
            if actions[i] == Positions.Short and action == Positions.Long:
                shorts.append(i)
                short_prices.append(prices[i])
            if actions[i] == Positions.Long and action == Positions.Short:
                longs.append(i)
                long_prices.append(prices[i])
            action = actions[i]

    ax3.plot(longs, long_prices, 'o', color='green', markersize=3)
    ax3.plot(shorts, short_prices, 'o', color='red', markersize=3)
    ax3.set(
        xlabel="time", ylabel="price", title="green = longs, red = shorts on the price of the asset"
    )

    plt.show()
