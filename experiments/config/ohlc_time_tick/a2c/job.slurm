#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=a2c_ohlc_time_tick
#SBATCH --output=a2c_ohlc_time_tick.out
#SBATCH --error=a2c_ohlc_time_tick.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick/a2c/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick/a2c/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick/a2c/ex3.yaml"
echo "Finished multi-experiment"