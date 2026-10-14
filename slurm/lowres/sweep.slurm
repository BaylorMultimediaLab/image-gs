#!/bin/bash

#SBATCH --job-name=igs_lowres
#SBATCH --output=results/slurm-logs/%x.%A_%a.out
#SBATCH --error=results/slurm-logs/%x.%A_%a.err
#SBATCH --account lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00

# One training run of the 480p/720p equal-time sweep per array task (main.py takes ~50 s at 480p and ~105 s at 720p on an A100).
# The arguments are passed on to slurm/lowres/sweep.py, which maps the array index to a run; use the same arguments for every task of an array.
# Submit from the repository root, after slurm/setup.slurm (slurm/lowres/submit.sh does both):
#   n=$(python3 slurm/lowres/sweep.py --count <arguments>)
#   sbatch --array=0-$((n-1)) slurm/lowres/sweep.slurm <arguments>
# Arguments: --phase <name> (results/lowres/<name>), --budget 480p=40,720p=90 (seconds), --res, --arms, --images (see slurm/lowres/sweep.py --help)
# The log directory must exist before submitting: mkdir -p results/slurm-logs
set -eo pipefail  # No -u: module (Lmod) and venv activation read unset variables

source slurm/env.sh
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
python slurm/lowres/sweep.py --task "${SLURM_ARRAY_TASK_ID}" "$@"
