#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc_twitter
#SBATCH --output=dqn_ohlc_twitter.out
#SBATCH --error=dqn_ohlc_twitter.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_twitter/dqn/ex3.yaml"
echo "Finished multi-experiment"