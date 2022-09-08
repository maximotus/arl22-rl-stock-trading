#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=stock-trading-dqn-ohlc-twitter
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+twitter.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+twitter.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/dqn-ohlc+twitter.yaml"
echo "Finished multi-experiment"