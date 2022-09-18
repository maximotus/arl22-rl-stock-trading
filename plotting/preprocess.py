from pathlib import Path
from plotting.utils import get_data_mem_save

import pandas as pd

results_dir = Path("./experiments/results")

norm_max = 182.84
norm_min = 130.01

test_data = get_data_mem_save(results_dir, "test")
test_data.to_csv("./test_data.csv")

train_data = get_data_mem_save(results_dir, "train")
train_data.to_csv("./train_data.csv")