#!/bin/bash
# Submit a real-time encoding job after building the environment (slurm/setup.slurm):
#   slurm/realtime/submit.sh sweep [slurm/realtime/sweep.py arguments]           time-to-HEVC sweep, an array of one task per rate point
#   slurm/realtime/submit.sh bench [phase] [slurm/realtime/bench.py arguments]   time per step and per frame (one job)
#   slurm/realtime/submit.sh profile [phase] [slurm/realtime/step_profile.py arguments]   wall-clock vs GPU time per step (one job)
#   e.g. slurm/realtime/submit.sh sweep --phase time-to-hevc
# Set SKIP_SETUP=1 to reuse an environment built earlier by slurm/setup.slurm (only if gsplat and compress/csrc have not changed since), with
# AFTER=<job id> to start only after that job succeeded (e.g., one setup job for several sweeps), and PARTITION=<name> to submit to another
# partition than the job's own (gpu_a100). Debug partitions (*_debug, e.g., gpu_a100_debug) take no job arrays, so there a sweep is one job
# that runs its tasks one after the other (~11 s each with the default budgets 0.5-4 s, as a warm-up encode adds 2 s).
set -eo pipefail

mode=${1:-}
if [ "${mode}" != "sweep" ] && [ "${mode}" != "bench" ] && [ "${mode}" != "profile" ]; then
    echo "Usage: slurm/realtime/submit.sh <sweep|bench|profile> [arguments]" >&2
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
elif [ -n "${AFTER:-}" ]; then
    dependency=(--dependency="afterok:${AFTER}")
fi
if [ "${mode}" = "sweep" ] && [[ "${PARTITION:-}" == *_debug ]]; then
    sbatch "${options[@]}" "${dependency[@]}" --time="$((5 + (num_tasks + 2) / 3))" slurm/realtime/sweep.slurm "$@"
    echo "Sweep of ${num_tasks} tasks submitted as one job (logs in results/slurm-logs)"
elif [ "${mode}" = "sweep" ]; then
    sbatch "${options[@]}" "${dependency[@]}" --array="0-$((num_tasks - 1))" slurm/realtime/sweep.slurm "$@"
    echo "Array of ${num_tasks} tasks submitted (logs in results/slurm-logs)"
elif [ "${mode}" = "bench" ]; then
    sbatch "${options[@]}" "${dependency[@]}" slurm/realtime/bench.slurm "$@"
    echo "Benchmark submitted (logs in results/slurm-logs, results in results/realtime/${1:-a100})"
else
    sbatch "${options[@]}" "${dependency[@]}" slurm/realtime/step_profile.slurm "$@"
    echo "Step profile submitted (logs in results/slurm-logs, results in results/realtime/${1:-a100-profile})"
fi
