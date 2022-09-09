#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=ppo_ohlc_twitter
#SBATCH --output=ppo_ohlc_twitter.out
#SBATCH --error=ppo_ohlc_twitter.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_twitter/ppo/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_twitter/ppo/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_twitter/ppo/ex3.yaml"
echo "Finished multi-experiment"