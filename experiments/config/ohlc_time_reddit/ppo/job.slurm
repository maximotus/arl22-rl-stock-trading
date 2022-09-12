#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=ppo_ohlc_time_reddit
#SBATCH --output=./logs/ohlc_time_reddit/ppo/ppo_ohlc_time_reddit.out
#SBATCH --error=./logs/ohlc_time_reddit/ppo/ppo_ohlc_time_reddit.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_reddit/ppo/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_reddit/ppo/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_reddit/ppo/ex3.yaml"
echo "Finished multi-experiment"