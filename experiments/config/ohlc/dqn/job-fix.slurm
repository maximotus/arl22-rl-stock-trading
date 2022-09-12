#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc
#SBATCH --output=dqn_ohlc.out
#SBATCH --error=dqn_ohlc.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc/dqn/ex3.yaml"
echo "Finished multi-experiment"