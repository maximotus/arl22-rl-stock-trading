#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=stock-trading-dqn-ohlc-reddit-twitter
#SBATCH --output=stock-trading-dqn-ohlc-reddit-twitter.out
#SBATCH --error=stock-trading-dqn-ohlc-reddit-twitter.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/a2c/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/a2c/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_time_tick_twitter_reddit/a2c/ex3.yaml"
echo "Finished multi-experiment"