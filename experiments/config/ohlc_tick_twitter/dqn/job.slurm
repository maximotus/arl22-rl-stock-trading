#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc_tick_twitter
#SBATCH --output=dqn_ohlc_tick_twitter.out
#SBATCH --error=dqn_ohlc_tick_twitter.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick_twitter/dqn/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick_twitter/dqn/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick_twitter/dqn/ex3.yaml"
echo "Finished multi-experiment"