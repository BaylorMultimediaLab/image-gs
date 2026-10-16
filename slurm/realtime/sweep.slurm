#!/bin/bash

#SBATCH --job-name=igs_realtime_sweep
#SBATCH --output=results/slurm-logs/%x.%A_%a.out
#SBATCH --error=results/slurm-logs/%x.%A_%a.err
#SBATCH --account lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00

# One task of the time-to-HEVC sweep per array task: one rate point of one image and arm at every time budget, ~75 s with the default
# budgets 0.5-32 s. The arguments are passed on to slurm/realtime/sweep.py, which maps the array index to a task; use the same arguments for
# every task of an array. Submit from the repository root, after slurm/setup.slurm (slurm/realtime/submit.sh sweep does both):
#   n=$(python3 slurm/realtime/sweep.py --count <arguments>)
#   sbatch --array=0-$((n-1)) slurm/realtime/sweep.slurm <arguments>
# Arguments: --phase <name> (results/realtime/<name>), --res, --images, --arms, --budgets, --set key=value (see slurm/realtime/sweep.py --help)
# Submitted without --array, the job runs all tasks one after the other, e.g., a small sweep on the debug partition:
#   sbatch --partition=gpu_a100_debug --time=00:30:00 slurm/realtime/sweep.slurm <arguments>
# The log directory must exist before submitting: mkdir -p results/slurm-logs
set -eo pipefail  # No -u: module (Lmod) and venv activation read unset variables

source slurm/env.sh
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
python slurm/realtime/sweep.py ${SLURM_ARRAY_TASK_ID:+--task "${SLURM_ARRAY_TASK_ID}"} "$@"
