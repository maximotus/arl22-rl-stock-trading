#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=stock-trading-dqn-ohlc-reddit-twitter
#SBATCH --output=stock-trading-dqn-ohlc-reddit-twitter.out
#SBATCH --error=stock-trading-dqn-ohlc-reddit-twitter.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+reddit+twitter.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+reddit+twitter.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+reddit+twitter.yaml"
echo "Finished multi-experiment"