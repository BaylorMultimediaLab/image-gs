#!/bin/bash
# Submit the 480p/720p equal-time sweep as a job array (one run per task) after building the environment.
# Usage (repository root, login node): slurm/submit_lowres.sh [slurm/lowres_sweep.py arguments]
#   e.g. slurm/submit_lowres.sh --phase equal-time-a100 --budget 480p=40,720p=90
# Set SKIP_SETUP=1 to reuse an environment built earlier by slurm/setup.slurm.
set -eo pipefail

mkdir -p results/slurm-logs
num_runs=$(python3 slurm/lowres_sweep.py --count "$@")
dependency=()
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    setup_job=$(sbatch --parsable slurm/setup.slurm)
    dependency=(--dependency="afterok:${setup_job%%;*}")
    echo "Setup job: ${setup_job%%;*}"
fi
sbatch "${dependency[@]}" --array="0-$((num_runs - 1))" slurm/lowres_sweep.slurm "$@"
echo "Array of ${num_runs} runs submitted (logs in results/slurm-logs)"
