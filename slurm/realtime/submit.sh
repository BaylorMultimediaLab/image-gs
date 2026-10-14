#!/bin/bash
# Submit a real-time encoding job after building the environment (slurm/setup.slurm):
#   slurm/realtime/submit.sh sweep [slurm/realtime/sweep.py arguments]           time-to-HEVC sweep, an array of one task per rate point
#   slurm/realtime/submit.sh bench [phase] [slurm/realtime/bench.py arguments]   time per step and per frame (one job)
#   e.g. slurm/realtime/submit.sh sweep --phase time-to-hevc
# Set SKIP_SETUP=1 to reuse an environment built earlier by slurm/setup.slurm (only if gsplat and compress/csrc have not changed since), and
# PARTITION=<name> to submit to another partition than gpu_a100 (e.g., gpu_a100_debug).
set -eo pipefail

mode=${1:-}
if [ "${mode}" != "sweep" ] && [ "${mode}" != "bench" ]; then
    echo "Usage: slurm/realtime/submit.sh <sweep|bench> [arguments]" >&2
    exit 1
fi
shift
mkdir -p results/slurm-logs
options=()
if [ -n "${PARTITION:-}" ]; then
    options+=(--partition="${PARTITION}")
fi
if [ "${mode}" = "sweep" ]; then
    num_tasks=$(python3 slurm/realtime/sweep.py --count "$@")
fi
dependency=()
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    setup_job=$(sbatch --parsable "${options[@]}" slurm/setup.slurm)
    dependency=(--dependency="afterok:${setup_job%%;*}")
    echo "Setup job: ${setup_job%%;*}"
fi
if [ "${mode}" = "sweep" ]; then
    sbatch "${options[@]}" "${dependency[@]}" --array="0-$((num_tasks - 1))" slurm/realtime/sweep.slurm "$@"
    echo "Array of ${num_tasks} tasks submitted (logs in results/slurm-logs)"
else
    sbatch "${options[@]}" "${dependency[@]}" slurm/realtime/bench.slurm "$@"
    echo "Benchmark submitted (logs in results/slurm-logs, results in results/realtime/${1:-a100})"
fi
