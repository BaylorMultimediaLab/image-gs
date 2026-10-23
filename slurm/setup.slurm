#!/bin/bash

#SBATCH --job-name=igs_setup
#SBATCH --output=results/slurm-logs/%x.%j.out
#SBATCH --error=results/slurm-logs/%x.%j.err
#SBATCH --account lp_h_opt_ga
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=18
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00

# Builds the image-gs environment once, so that array tasks only activate it: a venv on top of the PyTorch and SciPy modules, gsplat and
# fused-ssim compiled for the A100, and the LPIPS backbone weights. Then runs a smoke test and times optimization steps (slurm/lowres/calib.py).
# Submit from the repository root: sbatch slurm/setup.slurm  (slurm/<study>/submit.sh does this before a sweep)
set -eo pipefail  # No -u: module (Lmod) and venv activation read unset variables

if [ -z "${IGS_VENV}${VSC_SCRATCH}" ]; then
    echo "Set VSC_SCRATCH or IGS_VENV (location of the venv)" >&2
    exit 1
fi
source slurm/env.sh --no-activate
python -m venv --clear --system-site-packages "${VENV}"
source "${VENV}/bin/activate"
python -m pip install pip --upgrade

# requirements.txt pins the same versions as the local environment; the constraints keep the module builds of torch, numpy, and scipy
# (pip only installs a package from requirements.txt when the modules lack it, e.g., torchvision)
CONSTRAINTS=$(mktemp)
python -m pip list --format=freeze | grep -iE "^(torch|torchvision|numpy|scipy|pandas)==" > "${CONSTRAINTS}"
python -m pip install -c "${CONSTRAINTS}" -r requirements.txt --no-cache-dir

# CUDA extensions for the A100 (compute capability 8.0)
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS=${SLURM_CPUS_PER_GPU:-8}
BUILD_DIR=$(mktemp -d)
cp -r gsplat "${BUILD_DIR}/gsplat"
rm -rf "${BUILD_DIR}"/gsplat/build "${BUILD_DIR}"/gsplat/gsplat/csrc*.so  # A copied local build must not end up in the package
python -m pip install "${BUILD_DIR}/gsplat" -c "${CONSTRAINTS}" --no-cache-dir --no-build-isolation
# fused-ssim at the head of its pull request #46 and adds fused_l1_ssim (the training loss in one forward and one backward kernel)
python -m pip install "git+https://github.com/rahul-goel/fused-ssim/@77aba120314babc98140cd6c7e8c7119fa92cbee" -c "${CONSTRAINTS}" \
    --no-cache-dir --no-build-isolation
# C++ modules of compress/ (rANS coder and feature prediction), built from a copy as well
cp -r compress/csrc "${BUILD_DIR}/rans"
python -m pip install "${BUILD_DIR}/rans" -c "${CONSTRAINTS}" --no-cache-dir --no-build-isolation
rm -rf "${BUILD_DIR}" "${CONSTRAINTS}"

# LPIPS downloads its AlexNet backbone on first use; fetch it once instead of in every array task (main.py uses models/torch)
python -c "import torch; torch.hub.set_dir('models/torch'); import lpips; lpips.LPIPS(net='alex')"

python - <<'PY'
import torch, torchvision, numpy, scipy
from gsplat import csrc  # noqa: F401
from fused_ssim import fused_ssim  # noqa: F401
import compress.prediction  # noqa: F401  (checks that the C++ modules were built from compress/csrc)
import lpips, pytorch_msssim, flip_evaluator, skimage, cv2  # noqa: F401
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {numpy.__version__} | scipy {scipy.__version__} | "
      f"GPU {torch.cuda.get_device_name(0)}")
PY
mkdir -p results/lowres
python -m pip freeze --all > results/lowres/freeze_a100.txt  # Compare with the local environment (pip freeze)

# Smoke test: a short time-budgeted run that writes and verifies a bitstream, then the time per optimization step
python main.py --data_root=media/uvg --input_path=480p/Beauty_0001.png --num_gaussians=1000 --disable_prog_optim --max_steps=100000 \
    --max_time=5 --bitstream_codec=rans --color_transform=subtract-green --exp_name=slurm-smoke | tail -n 12
grep -q "Lossless decoding verified" results/slurm-smoke/*/log_train.txt
python slurm/lowres/calib.py results/lowres/calib_a100.csv \
    480p:1000:step,480p:4000:step,480p:4000:minmax-16-8-8-10,720p:9000:step,720p:9000:minmax-16-8-8-10
echo "Setup completed"
