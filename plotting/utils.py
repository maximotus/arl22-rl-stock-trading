from typing import List, Tuple
from pathlib import Path
from datetime import datetime

import os
import re
import json

import pandas as pd
import numpy as np
import tempfile


class InfDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "total_profit" in dct:
            dct["total_profit"] = float(f"{dct['total_profit']}")
        return dct


def create_data_frame(runs: List[Path]) -> pd.DataFrame:
    data = []
    for run in runs:
        item = pd.read_csv(run)
        item = item.drop(["Unnamed: 0"], axis=1)
        data.append(item)

    infos = []
    for i, item in enumerate(data):
        info = pd.DataFrame(list(map(fix_dict, item["info"].tolist())))
        other = item.drop(["info"], axis=1)
        for col in other.columns.tolist():
            info[col] = other[col]
        info["Run"] = i
        infos.append(info)
    return pd.concat(infos)


def fix_dict(ds: str) -> dict:
    fixed = re.sub(r"('position':) <Positions.[a-zA-Z]*: ([0-9])>", r"\1 \2", ds)
    fixed = re.sub(r"'", r'"', fixed)
    fixed = re.sub(r"([-]*)(inf)", r'"\1\2"', fixed)
    fixed_dict = json.loads(fixed, cls=InfDecoder)
    observation = fixed_dict.pop("observation")
    return {**fixed_dict, **observation}


def format(date: str):
    split = date.split("-")
    dt = datetime(
        int(split[0]),
        int(split[1]),
        int(split[2]),
        int(split[3]),
        int(split[4]),
        int(split[5]),
    )
    return int(dt.timestamp())


def get_paths(results_dir: Path):
    out = {}
    for attributes in os.listdir(results_dir):
        attributes_path = os.path.join(results_dir, attributes)
        attribute_results = {}
        for agent in os.listdir(attributes_path):
            agent_results = {"train": [], "test": []}
            agent_path = os.path.join(attributes_path, agent)
            for inter in os.listdir(agent_path):
                inter_path = os.path.join(agent_path, inter)
                for experiment in os.listdir(inter_path):
                    experiment_path = os.path.join(inter_path, experiment)
                    dates = os.listdir(experiment_path)
                    dates_dt = list(map(format, dates))
                    argmax = np.argmax(dates_dt)
                    latest = dates[argmax]
                    latest_path = os.path.join(experiment_path, latest)
                    results_train_path = os.path.join(
                        latest_path, "stats", "train-env", "result.csv"
                    )
                    results_test_path = os.path.join(
                        latest_path, "stats", "test-env", "result.csv"
                    )
                    agent_results["train"].append(results_train_path)
                    agent_results["test"].append(results_test_path)
            attribute_results[agent] = agent_results
        out[attributes] = attribute_results
    return out


def prepare_intermediate_df(data, setting, agent) -> pd.DataFrame:
    info = create_data_frame(data[setting])
    info = info.drop(["observation", "state"], axis=1)
    info["agent"] = agent
    return info


def get_data(result_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_paths(result_dir)
    print("Got Paths")

    all_attrs = {"train": None, "test": None}
    for attr, attr_vals in paths.items():
        agent_infos = {"train": None, "test": None}
        for agent, settings in attr_vals.items():
            # train_info = prepare_intermediate_df(settings, "train", agent)
            # agent_infos["train"].append(train_info)

            test_info = prepare_intermediate_df(settings, "test", agent)

            if agent_infos["test"] is None:
                agent_infos["test"] = test_info
            else:
                agent_infos["test"] = pd.concat([agent_infos["test"], test_info])
            # agent_infos["test"].append(test_info)
            print(f"Agent {agent} done")
        # train_data = pd.concat(agent_infos["train"])
        # train_data["attributes"] = attr

        test_data = agent_infos["test"]
        test_data["attributes"] = attr

        # all_attrs["train"].append(train_data)

        if all_attrs["test"] is None:
            all_attrs["test"] = test_data
        else:
            all_attrs["test"] = pd.concat([all_attrs["test"], test_data])
        # all_attrs["test"].append(test_data)
        print(f"Attribute {attr} done")
    return pd.concat(all_attrs["train"]), pd.concat(all_attrs["test"])


def get_data_mem_save(result_dir: Path, type: str):
    paths = get_paths(result_dir)
    print("Got Paths.")

    tmp_dir = tempfile.TemporaryDirectory()
    for attr, attr_vals in paths.items():
        print(f"...{attr}")
        tmp_attr = os.path.join(tmp_dir.name, f"{attr}.csv")

        agents_info = None
        for agent, settings in attr_vals.items():
            print(f"   ...{agent}")
            test_info = prepare_intermediate_df(settings, type, agent)
            test_info["time"] = test_info["time"].apply(str)
            if agents_info is None:
                agents_info = test_info
            else:
                agents_info = pd.concat([agents_info, test_info])
        agents_info["attributes"] = attr
        agents_info.to_csv(tmp_attr)

    print("...combine")
    all = None
    for file in os.listdir(tmp_dir.name):
        print(f"   ...{file}")
        inter = pd.read_csv(os.path.join(tmp_dir.name, file))
        if all is None:
            all = inter
        else:
            all = pd.concat([all, inter])
    tmp_dir.cleanup()
    print("done")
    return all


def rescale_column_minmax(
    data: pd.DataFrame, norm_min: float, norm_max: float, col: str
):
    out = data.copy()
    out[col] = out[col] * (norm_max - norm_min) + norm_min
    return out


def new_path(*args):
    path = os.path.join(*args)
    os.makedirs(path, exist_ok=True)
    return path
