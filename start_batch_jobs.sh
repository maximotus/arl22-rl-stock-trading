echo "Starting ohlc_time batch jobs (#16)"
sbatch experiments/config/ohlc_time/a2c/job.slurm
sbatch experiments/config/ohlc_time/dqn/job.slurm
sbatch experiments/config/ohlc_time/ppo/job.slurm

echo "Starting ohlc_tick_twitter batch jobs (#22)"
sbatch experiments/config/ohlc_tick_twitter/a2c/job.slurm
sbatch experiments/config/ohlc_tick_twitter/dqn/job.slurm
sbatch experiments/config/ohlc_tick_twitter/ppo/job.slurm

echo "Starting ohlc_time_reddit batch jobs (#23)"
sbatch experiments/config/ohlc_time_reddit/a2c/job.slurm
sbatch experiments/config/ohlc_time_reddit/dqn/job.slurm
sbatch experiments/config/ohlc_time_reddit/ppo/job.slurm

echo "Starting ohlc_time_tick_reddit batch jobs (#24)"
sbatch experiments/config/ohlc_time_tick_reddit/a2c/job.slurm
sbatch experiments/config/ohlc_time_tick_reddit/dqn/job.slurm
sbatch experiments/config/ohlc_time_tick_reddit/ppo/job.slurm

echo "Starting ohlc_time_twitter batch jobs (#28)"
sbatch experiments/config/ohlc_time_twitter/a2c/job.slurm
sbatch experiments/config/ohlc_time_twitter/dqn/job.slurm
sbatch experiments/config/ohlc_time_twitter/ppo/job.slurm
