#!/bin/bash
# Time-to-HEVC sweep of YUV features with luma weighting on the A100: one phase per variant, 24 tasks each (480p and 720p, QP 22-37, budgets
# 0.5-4 s), with the defaults otherwise (feat_prediction included). Variants:
#   yuv-rgb          RGB features (the sweep arms pin color_space rgb and subtract-green; the reference on the same code)
#   yuv-w1           YUV features with the RGB objective (uv_weight 1) and luma-equivalent steps (uv_step_ratio 1): should match yuv-rgb
#   yuv-w<w>-r<r>    uv_weight w and uv_step_ratio r (r1: the U and V steps that give them the error of RGB features); w0.25-r2 is the
#                    default, which had x265's 4:2:0 chroma PSNR in the local sweep (ycc-local-*), larger ratios gained little VMAF for
#                    much worse chroma
# Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise), which only needs rebuilding when gsplat or
# compress/csrc change (as with the C++ feature prediction, compress/csrc/prediction.cpp: submit once with SKIP_SETUP=0).
# Usage (repository root, on the cluster): slurm/realtime/submit_yuv.sh [variant ...]   (default: yuv-rgb yuv-w1 yuv-w0.25-r2)
# Analysis (locally, with results/realtime/yuv-* copied back):
#   .venv/bin/python experiments/realtime/frame_luma.py yuv-rgb yuv-w1 yuv-w0.25-r2
#   .venv/bin/python experiments/init/compare.py yuv-rgb yuv-w1 yuv-w0.25-r2 --arm qp-lam4-mid-add25-weight \
#       --metrics psnr,psnr_y,psnr_uv,vmaf,vmaf_neg --anchor results/lowres/hevc_noinfo.csv
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- yuv-rgb yuv-w1 yuv-w0.25-r2
fi
common=(--res 480p,720p --arms qp-lam4-mid-add25-weight --rates 22,27,32,37 --budgets 0.5,1,2,4 --set cuda_graph=true)
yuv=(--set color_space=yuv)
for variant in "$@"; do
    case "${variant}" in
        yuv-rgb)
            extra=() ;;
        yuv-w1)
            extra=("${yuv[@]}" --set uv_weight=1 --set uv_step_ratio=1) ;;
        yuv-w*-r*)
            weight=${variant#yuv-w}
            ratio=${weight#*-r}
            weight=${weight%-r*}
            extra=("${yuv[@]}" --set "uv_weight=${weight}" --set "uv_step_ratio=${ratio}") ;;
        *)
            echo "Unknown variant '${variant}'" >&2
            exit 1 ;;
    esac
    echo "Phase ${variant}:"
    SKIP_SETUP=${SKIP_SETUP:-1} slurm/realtime/submit.sh sweep --phase "${variant}" "${common[@]}" "${extra[@]}"
done
