from datetime import datetime
from importlib.resources import path
import os
import re
import json

from pathlib import Path
from typing import List
from webbrowser import get

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class InfDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "total_profit" in dct:
            dct['total_profit'] = float(f"{dct['total_profit']}")
        return dct


def plot_results(results_dir: Path, out_dir: Path):
    paths = __get_paths(results_dir)
    for attribute, attr_val in paths.items():
        for agent, values in attr_val.items():
            train_paths = values["train"]
            test_paths = values["test"]
            create_result_plots("train", attribute, agent, train_paths, out_dir)
            create_result_plots("test", attribute, agent, test_paths, out_dir)


def __get_paths(results_dir: Path):
    out = {}
    for attributes in os.listdir(results_dir):
        attributes_path = os.path.join(results_dir, attributes)
        attribute_results = {}
        for agent in os.listdir(attributes_path):
            agent_results = {
                "train": [],
                "test": []
            }
            agent_path = os.path.join(attributes_path, agent)
            for inter in os.listdir(agent_path):
                inter_path = os.path.join(agent_path, inter)
                for experiment in os.listdir(inter_path):
                    experiment_path = os.path.join(inter_path, experiment)
                    dates = os.listdir(experiment_path)
                    dates_dt = list(map(__format, dates))
                    argmax = (np.argmax(np.ndarray(dates_dt)))
                    latest = dates[argmax]
                    latest_path = os.path.join(experiment_path, latest)
                    results_train_path = os.path.join(latest_path, "stats", "train-env", "result.csv")
                    results_test_path = os.path.join(latest_path, "stats", "test-env", "result.csv")
                    agent_results["train"].append(results_train_path)
                    agent_results["test"].append(results_test_path)
            attribute_results[agent] = agent_results
        out[attributes] = attribute_results
    return out


def __format(date: str):
    split = date.split('-')
    dt = datetime(int(split[0]), int(split[1]), int(split[2]), int(split[3]), int(split[4]), int(split[5]))
    return int(dt.timestamp())


def create_result_plots(type: str, attr: str, agent: str, paths: List[Path], out_dir: Path):
    _, infos = __create_data_frames(paths)
    plot_mean_std_total_reward(type, attr, agent, out_dir, infos)
    plot_mean_std_total_profit(type, attr, agent, out_dir, infos)
    plot_per_run_comparison_total_reward(type, attr, agent, out_dir, infos)
    plot_per_run_comparison_total_profit(type, attr, agent, out_dir, infos)


def plot_mean_std_total_reward(type, attr, agent: str, out_dir, infos: pd.DataFrame):
    out_path = os.path.join(out_dir, attr, agent, type)
    os.makedirs(out_path, exist_ok=True)
    reward = sns.lineplot(x="time", y="total_reward", data=infos, errorbar="sd")
    reward.set(ylabel="Total Reward", xlabel="Time")

    sns.despine()
    plt.savefig(os.path.join(out_path, "reward_agg.png"))
    plt.clf()


def plot_mean_std_total_profit(type, attr, agent: str, out_dir, infos: pd.DataFrame):
    out_path = os.path.join(out_dir, attr, agent, type)
    os.makedirs(out_path, exist_ok=True)
    reward = sns.lineplot(x="time", y="total_profit", data=infos, errorbar="sd")
    reward.set(ylabel="Total Profit", xlabel="Time")

    sns.despine()
    plt.savefig(os.path.join(out_path, "profit_agg.png"))
    plt.clf()


def plot_per_run_comparison_total_reward(type, attr, agent: str, out_dir, infos: pd.DataFrame):
    out_path = os.path.join(out_dir, attr, agent, type)
    os.makedirs(out_path, exist_ok=True)
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
    plt.savefig(os.path.join(out_path, "reward_comp.png"))
    plt.clf()


def plot_per_run_comparison_total_profit(type, attr, agent: str, out_dir, infos: pd.DataFrame):
    out_path = os.path.join(out_dir, attr, agent, type)
    os.makedirs(out_path, exist_ok=True)
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
    plt.savefig(os.path.join(out_path, "profit_comp.png"))
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

plot_results("./experiments/results", "./result_plots/")
