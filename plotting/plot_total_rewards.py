import argparse
import re
import ast
import json

from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


class InfDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        # print(dct)
        if "total_profit" in dct:
            # print('here', dct['total_profit'])
            # dct['total_profit'] = float(f'{dct["total_profit"]}')
            dct['total_profit'] = float(f"{dct['total_profit']}")
        # if 'Actor' in dct:
        #     actor = Actor(dct['Actor']['Name'], dct['Actor']['Age'], '')
        #     movie = Movie(dct['Movie']['Title'], dct['Movie']['Gross'], '', dct['Movie']['Year'])
        #     return Edge(actor, movie)
        return dct


# TODO:
#   - create a Plot for the total reward over all three runs
#   - create a Plot for the AGGREGATED total reward over all three runs

#   - create a Plot for the total profit over all three runs
#   - create a Plot for the AGGREGATED total profit over all three runs

#   - create a Plot comparing the total reward over all three agents,
#     with aggregated data over the runs per agent

#   - create a Plot with the close price per time + background color with
#     buy or sell action.

# Plots for a single agent:
def plots_singel_agent(agent: str, runs: List[Path]):
    _, infos = __create_data_frames(runs)
    infos = infos.reset_index()
    plot_mean_std_total_reward(agent, infos)
    plot_mean_std_total_profit(agent, infos)
    plot_per_run_comparison_total_reward(agent, infos)
    plot_per_run_comparison_total_profit(agent, infos)


def plot_mean_std_total_reward(agent: str, infos: pd.DataFrame):
    reward = sns.lineplot(x="time", y="total_reward", data=infos, errorbar="sd")
    reward.set(ylabel="Total Reward", xlabel="Time")

    sns.despine()
    plt.savefig(f"./{agent}_total_reward_agg.png")
    plt.clf()


def plot_mean_std_total_profit(agent: str, infos: pd.DataFrame):
    reward = sns.lineplot(x="time", y="total_profit", data=infos, errorbar="sd")
    reward.set(ylabel="Total Profit", xlabel="Time")

    sns.despine()
    plt.savefig(f"./{agent}_total_profit_agg.png")
    plt.clf()


def plot_per_run_comparison_total_reward(agent: str, infos: pd.DataFrame):
    tr = infos[["time", "total_reward", "Run"]]
    profit = sns.lineplot(
        x="time",
        y="total_reward",
        hue="Run",
        data=tr,
        errorbar="sd",
        palette=sns.color_palette(),
    )
    profit.set(ylabel="Total Reward", xlabel="Time")

    sns.despine()
    plt.savefig(f"./{agent}_total_reward_comp.png")
    plt.clf()


def plot_per_run_comparison_total_profit(agent: str, infos: pd.DataFrame):
    tr = infos[["time", "total_profit", "Run"]]
    profit = sns.lineplot(
        x="time",
        y="total_profit",
        hue="Run",
        data=tr,
        errorbar="sd",
        palette=sns.color_palette(),
    )
    profit.set(ylabel="Total Reward", xlabel="Time")

    sns.despine()
    plt.savefig(f"./{agent}_total_profit_comp.png")
    plt.clf()


# Plots for a specific run:
def plot_buy_sell_decision_on_close(agent: str, run: str, data: Path):
    pass


# Plots over all agents:
def plot_agent_comparison(agents: List[str], runs: List[List[Path]]):
    pass


def __create_data_frames(runs: List[Path]) -> pd.DataFrame:
    data = []
    for run in runs:
        item = pd.read_csv(run)
        item = item.drop(["Unnamed: 0"], axis=1)
        data.append(item)

    infos = []
    for i, item in enumerate(data):
        info = pd.DataFrame(list(map(__fix_dict, item["info"].tolist())))
        info["Run"] = i
        infos.append(info)
    return data, pd.concat(infos)


def __fix_dict(ds: str) -> dict:
    fixed = re.sub(r"('position':) <Positions.[a-zA-Z]*: ([0-9])>", r"\1 \2", ds)
    fixed = re.sub(r"'", r'"', fixed)
    fixed = re.sub(r"([-]*)(inf)", r'"\1\2"', fixed)
    # fixed = re.sub(r"inf", r'"inf"', fixed)
    # print(fixed)
    # fixed_inf = re.sub(r"([-]*)(inf)", math.inf, fixed)
    fixed_dict = json.loads(fixed, cls=InfDecoder)
    # fixed_dict = ast.literal_eval(fixed)
    observation = fixed_dict.pop("observation")
    return {**fixed_dict, **observation}


# plots_singel_agent("Debug", runs=["./plotting/data/result-1.csv", "./plotting/data/result-2.csv", "./plotting/data/result-3.csv"])
# plots_singel_agent("A2C", runs=[
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/a2c/train/ex1/2022-09-09-13-36-04/stats\\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/a2c/train/ex2/2022-09-09-21-18-36/stats\\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/a2c/train/ex3/2022-09-10-04-59-54/stats\\result.csv",
# ])
plots_singel_agent(
    "DQN",
    runs=[
        "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/dqn/train/ex1/2022-09-08-19-13-30/stats\\result.csv",
        "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/dqn/train/ex2/2022-09-09-00-53-22/stats\\result.csv",
        "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time/dqn/train/ex3/2022-09-09-06-31-24/stats\\result.csv",
    ],
)
# plots_singel_agent("DQN", runs=[
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_reddit/dqn/train/ex1/2022-09-08-19-18-39/stats\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_reddit/dqn/train/ex2/2022-09-09-00-35-32/stats\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_reddit/dqn/train/ex3/2022-09-09-05-49-37/stats\result.csv",
# ])
# plots_singel_agent("DQN", runs=[
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_twitter/dqn/train/ex1/2022-09-08-19-19-13/stats\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_twitter/dqn/train/ex2/2022-09-09-00-33-56/stats\result.csv",
#     "/home/b/blenninger/arl22-rl-stock-trading/experiments/results/ohlc_time_twitter/dqn/train/ex3/2022-09-09-05-49-37/stats\result.csv",
# ])
