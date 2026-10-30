#!/bin/bash
#SBATCH --job-name=igs_sweep
#SBATCH --output=results/slurm-logs/%x.%j.out
#SBATCH --account=lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --time=01:00:00

# Runs main.py once per combination of the arguments, one after the other. Comma-separated values span a grid (here 3 x 4 x 4 = 48 runs):
#   sbatch slurm/sweep.slurm --exp_name=my-experiment --realtime --cuda_graph --max_time=2,4,10,20 --qp=22,27,32,37 \
#       --input_path=uvg/480p/Beauty_0001.png,uvg/480p/Bosphorus_0001.png,uvg/480p/Jockey_0001.png
# slurm/submit.sh sets the time limit from the budgets.
set -eo pipefail

module load cluster/wice/gpu_a100 Python/3.13.1-GCCcore-14.2.0 PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl SciPy-bundle/2025.06-gfbf-2025a
source "${VSC_SCRATCH}/image-gs/venv_a100/bin/activate"

combinations=1
for arg in "$@"; do
    IFS=, read -ra values <<< "${arg}"
    combinations=$((combinations * ${#values[@]}))
done

for ((task = 0; task < combinations; task++)); do
    # The task index picks one value of each argument
    i=${task}
    args=()
    for arg in "$@"; do
        key=""
        if [[ ${arg} == --*=* ]]; then
            key=${arg%%=*}=
            arg=${arg#*=}
        fi
        IFS=, read -ra values <<< "${arg}"
        args+=("${key}${values[i % ${#values[@]}]}")
        i=$((i / ${#values[@]}))
    done
    python main.py "${args[@]}" || echo "Combination ${task} failed: ${args[*]}" >&2
done
