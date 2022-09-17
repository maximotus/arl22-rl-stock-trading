import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

from rltrading import Data
from rltrading import Config
from typing import Tuple

import pandas as pd
import numpy as np




load_dotenv()
fh_key = os.getenv("FINNHUB_API_KEY", None)
symbol = os.getenv("SYMBOL", "AAPL")

start = datetime(2022, 1, 1, 0, 0, 1)
end = datetime(2022, 1, 7, 23, 59, 59)
social_lookback = timedelta(days=1)

data_config = Config(
    symbol=symbol, from_=start, to=end, lookback=social_lookback, finnhub_api_key=fh_key
)
aapl_data = Data()

dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, "data", "eval")
os.makedirs(path, exist_ok=True)
start = time.time()
aapl_data.fetch(store=True, config=data_config, dir_path=path)
total = time.time() - start
print(f"DONE ({total})")

"""
Split data and normalize it
"""

def z_score_scaling(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = data.copy()
    scalars = []
    for column in out.columns.tolist():
        curr = out[column]
        mean = curr.mean()
        std = curr.std()
        out[column] = (curr - mean) / std
        scalars.append({"column": column, "mean": mean, "std": std})
    return out, pd.DataFrame(scalars)


def min_max_scaling(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = data.copy()
    scalars = []
    for column in out.columns.tolist():
        curr = out[column]
        cmin = curr.min()
        cmax = curr.max()
        out[column] = (curr - cmin) / (cmax - cmin)
        scalars.append({"column": column, "min": cmin, "max": cmax})
    return out, pd.DataFrame(scalars)


def min_max_scaling_test(data: pd.DataFrame, scalars: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for column in out.columns.tolist():
        curr = out[column]
        curr_scalars = scalars.loc[scalars["column"] == column]
        cmin = curr_scalars["min"].item()
        cmax = curr_scalars["max"].item()
        out[column] = (curr - cmin) / (cmax - cmin)
    return out


def z_score_scaling_test(data: pd.DataFrame, scalars: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for column in out.columns.tolist():
        curr = out[column]
        curr_scalars = scalars.loc[scalars["column"] == column]
        mean = curr_scalars["mean"].item()
        std = curr_scalars["std"].item()
        out[column] = (curr - mean) / std
    return out


# Path to the raw data.
cwd = os.getcwd()
data_path = os.path.join(cwd, "experiments", "data", "eval", "AAPL.csv")

# Specify the sizes for the testing and the training split in percent.
training_split = 0.8

raw_data = pd.read_csv(data_path)
data_length = raw_data.shape[0]

train_length = int(np.ceil(data_length * training_split))
test_length = data_length - train_length

training_data = raw_data.iloc[:train_length]
testing_data = raw_data.iloc[train_length:]

# This is not nice code, however validating that the sizes are as expected
# might make debugging at a later stage easier.
assert training_data.shape[0] == train_length
assert testing_data.shape[0] == test_length

# NOW NORMALIZING
# MinMax Normalization
min_max_path = os.path.join(cwd, "experiments", "data", "minmax")
os.makedirs(min_max_path, exist_ok=True)

training_data_norm, scalars_norm = min_max_scaling(training_data)
testing_data_norm = min_max_scaling_test(testing_data, scalars_norm)

train_min_max_path = os.path.join(min_max_path, "train.csv")
test_min_max_path = os.path.join(min_max_path, "test.csv")
scalars_norm_path = os.path.join(min_max_path, "scalars.csv")

training_data_norm.to_csv(train_min_max_path, index=False)
testing_data_norm.to_csv(test_min_max_path, index=False)
scalars_norm.to_csv(scalars_norm_path, index=False)

# Z-Score Scaling
#zscore_path = os.path.join(cwd, "experiments", "data", "AAPL", "split", "zscore")
#os.makedirs(zscore_path, exist_ok=True)
#
#training_data_z, scalars_z = z_score_scaling(training_data)
#testing_data_z = z_score_scaling_test(testing_data, scalars_z)
#
#training_data_z_path = os.path.join(zscore_path, "train.csv")
#testing_data_z_path = os.path.join(zscore_path, "test.csv")
#scalars_z_path = os.path.join(zscore_path, "scalars.csv")
#
#training_data_z.to_csv(training_data_z_path, index=False)
#testing_data_z.to_csv(testing_data_z_path, index=False)
#scalars_z.to_csv(scalars_z_path, index=False)


# nur trainings daten normalisieren
