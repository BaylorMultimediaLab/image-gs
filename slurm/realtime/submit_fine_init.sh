#!/bin/bash
# Time-to-HEVC sweep of the fine-blob initialization (init_fine_ratio) on the A100, with the init study's settings (init_scale_knn 0.4,
# init_aspect 4, add_color_init target, the defaults): one phase per ratio, fine-ref for 0 (the reference on the same code) and
# fine-<ratio> otherwise, 24 tasks each (480p and 720p, QP 22-37, budgets 0.5-4 s). Reuses the environment built by slurm/setup.slurm
# (SKIP_SETUP=1 unless set otherwise), which only needs rebuilding when gsplat or compress/csrc change.
# Usage (repository root, on the cluster): slurm/realtime/submit_fine_init.sh [ratio ...]   (default: 0 0.2 0.3 0.4)
# Analysis (locally, with results/realtime/fine-* copied back):
#   .venv/bin/python experiments/realtime/frame_luma.py fine-ref fine-0.2 fine-0.3 fine-0.4
#   .venv/bin/python experiments/init/compare.py fine-ref fine-0.2 fine-0.3 fine-0.4 --arm qp-lam4-mid-add25-weight \
#       --metrics psnr,psnr_y,vmaf,vmaf_neg --anchor results/lowres/hevc_noinfo.csv
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- 0 0.2 0.3 0.4
fi
common=(--res 480p,720p --arms qp-lam4-mid-add25-weight --rates 22,27,32,37 --budgets 0.5,1,2,4 --set cuda_graph=true
        --set init_scale_knn=0.4 --set init_aspect=4 --set add_color_init=target)
for ratio in "$@"; do
    phase="fine-${ratio}"
    if [ "${ratio}" = "0" ]; then
        phase=fine-ref
    fi
    extra=(--set "init_fine_ratio=${ratio}")
    echo "Phase ${phase}:"
    SKIP_SETUP=${SKIP_SETUP:-1} slurm/realtime/submit.sh sweep --phase "${phase}" "${common[@]}" "${extra[@]}"
done
