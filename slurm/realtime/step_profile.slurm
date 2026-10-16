#!/bin/bash

#SBATCH --job-name=igs_step_profile
#SBATCH --output=results/slurm-logs/%x.%j.out
#SBATCH --error=results/slurm-logs/%x.%j.err
#SBATCH --account lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00

# Profile of an optimization step in real-time mode (slurm/realtime/step_profile.py): wall-clock vs GPU busy time per step, i.e., what CUDA
# graphs can save, plus GPU time per kernel and CPU time per phase. A single job (no array), so it runs on the debug partition; a few minutes.
# Uses the environment built by slurm/setup.slurm (no rebuild needed, the script is Python only). Submit from the repository root:
#   mkdir -p results/slurm-logs && sbatch slurm/realtime/step_profile.slurm [phase] [slurm/realtime/step_profile.py arguments]
# Results go to results/realtime/<phase> (default a100-profile).
set -eo pipefail  # No -u: module (Lmod) and venv activation read unset variables

phase=${1:-a100-profile}
source slurm/env.sh
nvidia-smi --query-gpu=name,clocks.max.sm --format=csv,noheader
lscpu | grep "Model name"
python slurm/realtime/step_profile.py --phase "${phase}" "${@:2}"
