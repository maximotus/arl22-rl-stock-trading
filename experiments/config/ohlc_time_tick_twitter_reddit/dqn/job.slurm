#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc_time_tick_twitter_reddit
#SBATCH --output=dqn_ohlc_time_tick_twitter_reddit.out
#SBATCH --error=dqn_ohlc_time_tick_twitter_reddit.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/dqn/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/dqn/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/dqn/ex3.yaml"
echo "Finished multi-experiment"