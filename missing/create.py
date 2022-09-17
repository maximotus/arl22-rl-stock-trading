import os

attrs = [
    "ohlc",
    # "ohlc_reddit",
    "ohlc_twitter",
    # "ohlc_tick_reddit",
    "ohlc_reddit_twitter",
]
agents = [
    "a2c", "ppo", "dqn"
]
experiments = ["ex1", "ex2", "ex3"]

items = []
for attr in attrs:
    for agent in agents:
        for exp in experiments:
            items.append((attr, agent, exp))

print(items)
print(len(items))

cwd = os.getcwd()

for attr, agent, ex in items:
    slurm_job_name = f"{agent}_{attr}_{ex}.slurm"
    slurm_job_path = os.path.join(cwd, slurm_job_name)

    logs_path = os.path.join("missing", "logs", attr, agent, ex)

    ex_main_path = os.path.join("experiments", "main.py")
    conf_path = os.path.join("experiments", "config", attr, agent, f"{ex}.yaml")

    with open(slurm_job_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write(f'#SBATCH --nodes=1\n')
        f.write(f'#SBATCH --job-name={agent}_{attr}_{ex}\n')
        f.write(f'#SBATCH --output={os.path.join(logs_path, "log.out")}\n')
        f.write(f'#SBATCH --error={os.path.join(logs_path, "log.err")}\n')
        f.write(f'#SBATCH --partition=All\n')
        f.write(f'echo "Starting missing-experiment..."\n')
        f.write(f'poetry run python {ex_main_path} --conf "{conf_path}"\n')
        f.write(f'echo "Finished missing-experiment"\n')
    
    run_all_path = os.path.join(cwd, "run_missing.sh")
    with open(run_all_path, "a") as f:
        f.write(f"sbatch {slurm_job_path}\n")
