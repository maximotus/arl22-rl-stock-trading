import os
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


buy_sell_cmap = ["#EC7063", "#52BE80"]


# SINGLE AGENT


def plot_agent_runs_sd(
    data: pd.DataFrame, y_value: str, y_title: str, agent: str, attr: str, out_dir: Path
):
    sns.set(rc={"figure.figsize": (10, 6.25)})
    profit = sns.lineplot(
        x="time", y=y_value, errorbar="sd", data=data, palette="Dark2"
    )
    profit.set(ylabel=y_title, xlabel="Time")
    profit.set_xticks([])
    plt.title(label=f"{y_title} over all runs of the '{agent}' agent")
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"{agent}_{attr}_{y_value}_runs_sd.png"))
    plt.clf()


def plot_buy_sell_decision_run_comparison(
    data: pd.DataFrame, attributes: str, agent: str, out_dir: Path
):
    sns.set(rc={"figure.figsize": (16, 2)})
    decisions = data[["position", "time", "Run"]]
    decisions_plt = decisions.pivot("Run", "time", "position")
    ax = sns.heatmap(decisions_plt, cmap=buy_sell_cmap)

    # Fix color bar
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + r / 2 * (0.5 + i) for i in range(2)])
    colorbar.set_ticklabels(["Short", "Long"])
    ax.set_xticks([])
    plt.title(
        label=f"Positions at each time step for the '{agent}' on the '{attributes}' data"
    )
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"positions_run_comp_{agent}_{attributes}.png"))
    plt.clf()


def plot_buy_sell_decision_detail(
    data: pd.DataFrame, y_value: str, y_title: str, agent: str, attr: str, out_dir: Path
):
    def __plt(d: pd.DataFrame, run: int):
        positions = d["position"].tolist()
        times = d["time"].tolist()
        pos_times = zip(positions, times)
        for position, time in pos_times:
            line = plt.axvline(time, color=buy_sell_cmap[position])
            line.set(alpha=0.04)
        course = sns.lineplot(d, x="time", y=y_value, color="black")
        course.set_xticks([])
        sns.despine()
        plt.savefig(os.path.join(out_dir, f"positions_{agent}_{attr}_{run}_{y_value}.png"))
        plt.clf()

    data["time"] = data["time"].apply(str)
    runs = pd.unique(data["Run"]).tolist()
    for run in runs:
        run_data = data.loc[data["Run"] == run]
        __plt(run_data, run)


def plot_attribute_mean_agent_comparison(
    data: pd.DataFrame, y_value: str, y_title: str, agent: str, out_dir: Path
):
    data_plt = data.loc[data["agent"] == agent]
    data_plt = data[["time", "total_profit", "Run", "attributes"]]
    data_plt = (
        data.groupby(["attributes", "time"])
        .mean()
        .reset_index()[["time", "total_profit", "attributes"]]
    )

    sns.set(rc={"figure.figsize": (10, 6.25)})
    profit = sns.lineplot(
        x="time", y=y_value, hue="attributes", data=data_plt, palette="Dark2"
    )
    profit.set(ylabel="Total Profit", xlabel="Time")
    profit.set_xticks([])
    plt.legend(loc="upper left", title="Attributes", bbox_to_anchor=(1.02, 1))
    plt.title(label=f"{y_title} over all evaluated attributes for the '{agent}' agent")
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"{agent}_{y_value}_mean_run.png"))
    plt.clf()


# SINGLE ATTRIBUTE


def plot_attribute_agent_comparison(
    data: pd.DataFrame, y_value: str, y_title: str, attribute: str, out_dir: Path
):
    sns.set(rc={"figure.figsize": (10, 6.25)})
    profit = sns.lineplot(
        x="time", y=y_value, hue="agent", errorbar="sd", data=data, palette="Dark2"
    )
    profit.set(ylabel=y_title, xlabel="Time")
    profit.set_xticks([])
    plt.legend(loc="upper left", title="Agent")
    plt.title(label=f"Per Agent comparison of the {y_title} for the '{attribute}' data")
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"{attribute}_{y_value}_agent_comp.png"))
    plt.clf()


# OVERALL


def plot_overall_comparison(
    data: pd.DataFrame, y_value: str, y_title: str, out_dir: Path
):
    sns.set(rc={"figure.figsize": (10, 6.25)})
    profit = sns.lineplot(
        x="time", y=y_value, hue="attributes", style="agent", data=data, palette="Dark2"
    )
    profit.set(ylabel="Total Profit averaged over three runs", xlabel="Time")
    profit.set_xticks([])
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.title(label=f"{y_title} over all evaluated attributes over all agents")
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"overall_comp.png"))
    plt.clf()


def plot_final_profit_heatmap(data: pd.DataFrame, out_dir: Path):
    sns.set(rc={"figure.figsize": (10, 2)})
    heat_data = data[["time", "total_profit", "attributes", "agent", "Run"]]
    max_time = data["time"].max()
    heat_data = heat_data.loc[heat_data["time"] == max_time]
    heat_data = heat_data[["total_profit", "attributes", "agent", "Run"]]
    avg_heat_data = (
        heat_data.groupby(["attributes", "agent"])
        .mean()
        .reset_index()[["total_profit", "attributes", "agent"]]
    )
    avg_heat_data_plot = avg_heat_data.pivot("agent", "attributes", "total_profit")
    sns.set(rc={"figure.figsize": (10, 2)})
    sns.heatmap(avg_heat_data_plot, annot=True, linewidths=0.5, cmap="crest")
    plt.title(
        label="Total Profit averaged over all runs for each agent/attribute combination"
    )
    sns.despine()
    plt.savefig(os.path.join(out_dir, f"average_total_profit_heatmap.png"))
    plt.clf()
