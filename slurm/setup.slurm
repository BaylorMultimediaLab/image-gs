#!/bin/bash
#SBATCH --job-name=igs_setup
#SBATCH --output=results/slurm-logs/%x.%j.out
#SBATCH --account=lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18
#SBATCH --time=01:00:00

# Builds the venv of the sweeps. From the repository root: mkdir -p results/slurm-logs && sbatch slurm/setup.slurm
set -eo pipefail

module load cluster/wice/gpu_a100 Python/3.13.1-GCCcore-14.2.0 PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl SciPy-bundle/2025.06-gfbf-2025a
venv=${VSC_SCRATCH}/image-gs/venv_a100
python -m venv --clear --system-site-packages "${venv}"
source "${venv}/bin/activate"
pip install --upgrade pip

# Keep the module builds of torch, numpy, and scipy
build=$(mktemp -d)
pip list --format=freeze | grep -iE "^(torch|torchvision|numpy|scipy|pandas)==" > "${build}/constraints.txt"
pip install -c "${build}/constraints.txt" -r requirements.txt --no-cache-dir

# CUDA and C++ extensions for the A100, built from clean copies
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=${SLURM_CPUS_PER_GPU}
cp -r gsplat "${build}/gsplat"
cp -r compress/csrc "${build}/rans"
rm -rf "${build}"/gsplat/build "${build}"/gsplat/gsplat/csrc*.so
pip install -c "${build}/constraints.txt" --no-cache-dir --no-build-isolation "${build}/gsplat" "${build}/rans" \
    "git+https://github.com/rahul-goel/fused-ssim/@77aba120314babc98140cd6c7e8c7119fa92cbee"
rm -rf "${build}"

# LPIPS weights, fetched once instead of in every run
python -c "import torch; torch.hub.set_dir('models/torch'); import lpips; lpips.LPIPS(net='alex')"

python main.py --exp_name=slurm-smoke --max_time=5
grep -rq "Lossless decoding verified" results/slurm-smoke
echo "Setup completed"
