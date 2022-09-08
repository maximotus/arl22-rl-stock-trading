import argparse
import re
import ast

from pathlib import Path
from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    profit = sns.lineplot(x="time", y="total_reward", hue="Run", data=tr, errorbar="sd", palette=sns.color_palette())
    profit.set(ylabel="Total Reward", xlabel="Time")

    sns.despine()
    plt.savefig(f"./{agent}_total_reward_comp.png")
    plt.clf()


def plot_per_run_comparison_total_profit(agent: str, infos: pd.DataFrame):
    tr = infos[["time", "total_profit", "Run"]]
    profit = sns.lineplot(x="time", y="total_profit", hue="Run", data=tr, errorbar="sd", palette=sns.color_palette())
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
    fixed_dict = ast.literal_eval(fixed)
    observation = fixed_dict.pop("observation")
    return {**fixed_dict, **observation}


plots_singel_agent("Debug", runs=["./plotting/data/result-1.csv", "./plotting/data/result-2.csv", "./plotting/data/result-3.csv"])
