#!/bin/bash
# Submits a real-time sweep over QPs, resolutions, UVG sequences, and time budgets as one job that runs the combinations one after the
# other; other arguments go to main.py, e.g.:
#   slurm/submit.sh --qps 22,27,32,37 --res 480p,720p --sequences Beauty,Bosphorus,Jockey --budgets 2,4,10,20 --exp_name=my-experiment \
#       --uv_weight=0.25,0.5
# The time limit is 1.5x the estimate: the sum of the budgets plus RUN_OVERHEAD_S (default 10) per run for start-up and saving
# (at most 1 h on gpu_a100_debug).
# SKIP_SETUP=0 builds the venv first (slurm/setup.slurm); PARTITION=gpu_a100_debug submits to the debug partition.
# Results (slurm/sweep.slurm) and Slurm logs go to $VSC_DATA/image-gs/<exp_name>, off the small home partition. The job is named after
# --exp_name.
set -eo pipefail

qps=22,27,32,37
res=480p,720p
sequences=Beauty,Bosphorus,Jockey
budgets=2,4,10,20
exp_name=test/uvg
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --qps) qps=$2; shift 2 ;;
        --res=*) res=${1#*=}; shift ;;
        --res) res=$2; shift 2 ;;
        --sequences=*) sequences=${1#*=}; shift ;;
        --sequences) sequences=$2; shift 2 ;;
        --budgets) budgets=$2; shift 2 ;;
        --exp_name=*) exp_name=${1#*=}; args+=("$1"); shift ;;
        --exp_name) exp_name=$2; args+=("$1" "$2"); shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done

args=(--realtime --cuda_graph --res="${res}" --sequences="${sequences}" --qp="${qps}" --max_time="${budgets}" "${args[@]}")

combinations=1
for arg in "${args[@]}"; do
    IFS=, read -ra values <<< "${arg}"
    combinations=$((combinations * ${#values[@]}))
done
# Every budget runs once per combination of the other arguments
estimate=$(echo "${budgets}" | tr ',' '\n' | awk -v runs="${combinations}" -v overhead="${RUN_OVERHEAD_S:-10}" \
    '{sum += $1; n += 1} END {print int((runs / n * sum + runs * overhead) / 60) + 1}')
minutes=$((estimate * 3 / 2))

partition=${PARTITION:-gpu_a100}
if [ "${partition}" = gpu_a100_debug ] && [ "${minutes}" -gt 60 ]; then
    echo "Warning: estimated ${estimate} min (${minutes} min with margin), but gpu_a100_debug allows at most 60 min" >&2
    minutes=60
fi
logs=${VSC_DATA}/image-gs/${exp_name}
mkdir -p results/slurm-logs "${logs}"
dependency=()
if [ "${SKIP_SETUP:-1}" = 0 ]; then
    setup=$(sbatch --parsable --partition="${partition}" slurm/setup.slurm)
    dependency=(--dependency="afterok:${setup%%;*}")
fi
sbatch "${dependency[@]}" --job-name="${exp_name//\//-}" --output="${logs}/%x.%j.out" --error="${logs}/%x.%j.err" --partition="${partition}" \
    --time="${minutes}" slurm/sweep.slurm "${args[@]}"
echo "${combinations} runs, estimated ${estimate} min, time limit ${minutes} min"
