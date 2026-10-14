#!/bin/bash

#SBATCH --job-name=igs_realtime
#SBATCH --output=results/slurm-logs/%x.%j.out
#SBATCH --error=results/slurm-logs/%x.%j.err
#SBATCH --account lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00

# Real-time encoding benchmark in one job, so that every timing gets the same idle GPU: the time per optimization step in default and
# real-time mode (slurm/lowres/calib.py, the method of results/lowres/calib_a100.csv), then the wall-clock time per frame of a persistent
# encoder process at a 1-second budget (slurm/realtime/bench.py). Results go to results/realtime/<phase>.
# Submit from the repository root, after slurm/setup.slurm (slurm/realtime/submit.sh bench does both):
#   sbatch slurm/realtime/bench.slurm <phase> [slurm/realtime/bench.py arguments]
# The log directory must exist before submitting: mkdir -p results/slurm-logs
set -eo pipefail  # No -u: module (Lmod) and venv activation read unset variables

phase=${1:-a100}
source slurm/env.sh
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
lscpu | grep "Model name"
mkdir -p "results/realtime/${phase}"
python slurm/lowres/calib.py "results/realtime/${phase}/calib.csv" \
    480p:1000:step,480p:1000:step:realtime,480p:4000:step,480p:4000:step:realtime,720p:9000:step,720p:9000:step:realtime
python slurm/realtime/bench.py --phase "${phase}" "${@:2}"
