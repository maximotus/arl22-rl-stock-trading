#!/bin/sh
#SBATCH --nodes=1
#SBATCH --job-name=dqn_ohlc_tick
#SBATCH --output=dqn_ohlc_tick.out
#SBATCH --error=dqn_ohlc_tick.err
#SBATCH --partition=All
echo "Starting multi-experiment..."
conda activate arl22
echo "Starting run 1..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick/dqn/ex1.yaml"
echo "Starting run 2..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick/dqn/ex2.yaml"
echo "Starting run 3..."
poetry run python experiments/main.py --conf "./experiments/config/ohlc_tick/dqn/ex3.yaml"
echo "Finished multi-experiment"