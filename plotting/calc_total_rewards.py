from turtle import position
import pandas as pd
from typing import List, Tuple
from plotting.utils import get_paths
from pathlib import Path
from rltrading.gym.environment import Positions, Actions
import  os
import re
import json

_result_dir = Path("./experiments/results")

norm_min: float = 130.01
norm_max: float = 182.84

def calc_total_rewards(result_dir: Path, type: str):
	paths = get_paths(result_dir)

	for attr, attr_vals in paths.items():
		print(f"...{attr}")

		agents_info = None
		for agent, settings in attr_vals.items():
			for setting in settings[type]:
				df = load_df(setting)
				calc(df)
				df.to_csv(setting)
	return all

class InfDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "total_profit" in dct:
            dct["total_profit"] = float(f"{dct['total_profit']}")
        return dct

def fix_dict(ds: str) -> dict:
    fixed = re.sub(r"('position':) <Positions.[a-zA-Z]*: ([0-9])>", r"\1 \2", ds)
    fixed = re.sub(r"'", r'"', fixed)
    fixed = re.sub(r"([-]*)(inf)", r'"\1\2"', fixed)
    fixed_dict = json.loads(fixed, cls=InfDecoder)
    return {**fixed_dict}
	
def load_df(run: Path):
	item = pd.read_csv(run)
	item = item.drop(["Unnamed: 0"], axis=1)

	items = {'info': list(map(fix_dict, item["info"].tolist()))}
	info = pd.DataFrame(items)
	item["info"] = info
	print(item)
	print(info)
	return item

def __rescale(val: float) -> float:
	return val * (norm_max - norm_min) + norm_min

def calc(df: pd.DataFrame): 
	last_trade_price = 0
	total_profit = 1
	action = 1
	for index, row in df.iterrows():
		action = row['action']
		info = row['info']
		curr_close = info['observation']['close']

		if index == 0:
			last_trade_price = info['observation']['close']
			position = info['position']

		if (position == Positions.Long) and (action == Actions.Sell.value):
			total_profit *= __rescale(curr_close) / __rescale(last_trade_price)
			last_trade_price = curr_close
			position = info['position']
			row['info']['total_profit'] = total_profit
			
		if (position == Positions.Short) and (action == Actions.Buy.value):
			total_profit *= __rescale(last_trade_price) / __rescale(curr_close)
			last_trade_price = curr_close
			position = info['position']
			row['info']['total_profit'] = total_profit


calc_total_rewards(_result_dir, 'test')