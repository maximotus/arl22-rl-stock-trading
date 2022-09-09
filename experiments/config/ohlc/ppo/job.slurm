#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=ppo_ohlc
#SBATCH --output=ppo_ohlc.out
#SBATCH --error=ppo_ohlc.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc/ppo/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc/ppo/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc/ppo/ex3.yaml"
echo "Finished multi-experiment"