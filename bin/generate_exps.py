import collections
import copy
import os 

log_dir = 'logs'
slurm_path = 'slurm_logs'

with open('bin/train_all_models.sh') as f:
    for ind, cmd in enumerate(f):
        common = f"""#!/bin/bash
#SBATCH --job-name=semantic-parsing-{ind}
#SBATCH -o {slurm_path}/logs_{ind}.out
#SBATCH -e {slurm_path}/logs_{ind}.err
#SBATCH --time=0-4:00:00
#SBATCH --partition=1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --exclude=node030"""

        with open(os.path.join(log_dir, f'script_{ind}.sh'), 'w') as f:
            f.write(common + "\n")
            f.write(cmd + "\n")

        with open('experiments.txt', 'w') as f:
            f.write(f'{log_dir}\n')