import os
from pathlib import Path

from plotting.utils import get_data_mem_save, rescale_column_minmax, new_path
from plotting.plots import (
    plot_agent_runs_sd,
    plot_attribute_mean_agent_comparison,
    plot_buy_sell_decision_run_comparison,
    plot_buy_sell_decision_detail,
    plot_attribute_agent_comparison,
    plot_overall_comparison,
    plot_final_profit_heatmap,
)

import pandas as pd

results_dir = Path("./experiments/results")
out_dir_base = Path("./experiments/plots")
norm_max = 182.84
norm_min = 130.01

# test_data = get_data_mem_save(results_dir, "test")
# test_data.to_csv("./test_data.csv")
test_data = pd.read_csv("./test_data.csv")
test_data["time"] = test_data["time"].apply(str)
test_data["close"] = test_data["close"] * (norm_max - norm_min) + norm_min

# TODO: same for test data

data = test_data
attributes = pd.unique(data["attributes"]).tolist()
agents = pd.unique(data["agent"]).tolist()

# SINGLE AGENT PLOTS
# - plot_agent_runs_sd
# - plot_attribute_mean_agent_comparison
# - plot_buy_sell_decision_run_comparison
# - plot_buy_sell_decision_detail
for attribute in attributes:
    attr_data = data.loc[data["attributes"] == attribute]

    out_dir = new_path(out_dir_base, attribute)
    plot_attribute_agent_comparison(
        attr_data, "total_profit", "Total Profit", attribute, out_dir
    )
    plot_attribute_agent_comparison(
        attr_data, "total_reward", "Total Reward", attribute, out_dir
    )

    for agent in agents:
        agent_data = attr_data.loc[attr_data["agent"] == agent]
        out_dir = new_path(out_dir_base, attribute, agent)
        plot_agent_runs_sd(
            agent_data, "total_profit", "Total Profit", agent, attribute, out_dir
        )
        plot_agent_runs_sd(
            agent_data, "total_reward", "Total Reward", agent, attribute, out_dir
        )
        plot_buy_sell_decision_run_comparison(agent_data, attribute, agent, out_dir)
        plot_buy_sell_decision_detail(
            agent_data, "total_profit", "Total Profit", agent, attribute, out_dir
        )
        plot_buy_sell_decision_detail(
            agent_data, "total_reward", "Total Reward", agent, attribute, out_dir
        )
        plot_buy_sell_decision_detail(
            agent_data, "close", "Close Price", agent, attribute, out_dir
        )

for agent in agents:
    agent_data = data.loc[data["agent"] == agent]
    out_dir = new_path(out_dir_base, agent)
    plot_attribute_mean_agent_comparison(
        agent_data, "total_profit", "Total Profit", agent, out_dir
    )
    plot_attribute_mean_agent_comparison(
        agent_data, "total_reward", "Total Reward", agent, out_dir
    )

# OVERALL COMPARISON
# - plot_overall_comparison
# - plot_final_profit_heatmap
out_dir = out_dir_base
plot_overall_comparison(data, "total_profit", "Total Profit", out_dir)
plot_overall_comparison(data, "total_reward", "Total Reward", out_dir)

plot_final_profit_heatmap(data, out_dir)
plot_final_profit_heatmap(data, out_dir)
