# Environment of the image-gs jobs on wICE (A100): toolchain modules and the venv built by slurm/setup.slurm.
# Source it from the repository root: source slurm/env.sh
module load cluster/wice/gpu_a100
module load Python/3.13.1-GCCcore-14.2.0
module load PyTorch-bundle/2.9.1-foss-2025a-CUDA-12.8.0-whl
module load SciPy-bundle/2025.06-gfbf-2025a

VENV=${IGS_VENV:-${VSC_SCRATCH}/image-gs/venv_a100}
if [ "${1:-}" != "--no-activate" ]; then
    source "${VENV}/bin/activate"
fi
