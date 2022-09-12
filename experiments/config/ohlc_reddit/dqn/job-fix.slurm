#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc_reddit
#SBATCH --output=dqn_ohlc_reddit.out
#SBATCH --error=dqn_ohlc_reddit.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_reddit/dqn/ex3.yaml"
echo "Finished multi-experiment"