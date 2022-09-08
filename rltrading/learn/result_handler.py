import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from rltrading.gym.environment import Actions, Positions


def plot_result(result_memory, save_path=None):
    time = range(len(result_memory))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)
    # fig, axes = plt.subplots(2, 2, figsize=(20,8))
    plt.figure(figsize=(15, 8))

    # print reward curve
    ax = plt.subplot(gs[0, 0])
    res = list(map(lambda x: x.reward, result_memory))
    cum_rewards = np.cumsum(res)
    ax.plot(time, cum_rewards)
    ax.set(
        xlabel="time", ylabel="cumulative rewards", title="cumulative rewards over time"
    )
    ax.grid()

    # print action count
    ax2 = plt.subplot(gs[0, 1])
    actions = list(map(lambda x: x.action, result_memory))
    print("action 0: ", actions.count(0), " actions 1: ", actions.count(1))
    labels, counts = np.unique(actions, return_counts=True)
    ax2.bar(labels, counts, align="center")
    plt.gca().set_xticks(labels)
    ax2.set(
        xlabel="action",
        ylabel="n times chosen",
        title="the amount each action got chosen",
    )

    # print where shorts and longs were taken
    ax3 = plt.subplot(gs[1, :])
    prices = list(
        map(lambda x: x.observation[len(x.observation) - 1]["close"], result_memory)
    )
    ax3.plot(time, prices)
    actions = list(map(lambda x: x.action, result_memory))
    action = actions[0]
    longs = []
    long_prices = []
    shorts = []
    short_prices = []
    for i in range(1, len(actions)):
        if actions[i] != action:
            if actions[i] == Positions.Short and action == Positions.Long:
                shorts.append(i)
                short_prices.append(prices[i])
            if actions[i] == Positions.Long and action == Positions.Short:
                longs.append(i)
                long_prices.append(prices[i])
            action = actions[i]

    ax3.plot(longs, long_prices, "o", color="green", markersize=3)
    ax3.plot(shorts, short_prices, "o", color="red", markersize=3)
    ax3.set(
        xlabel="time",
        ylabel="price",
        title="green = longs, red = shorts on the price of the asset",
    )

    plt.show()

    if save_path:
        plt.savefig(save_path + "\\result_graph.png")


def save_result(result_memory, save_path: str):
    path = save_path + "\\result.csv"
    pd.DataFrame(result_memory).to_csv(path)
