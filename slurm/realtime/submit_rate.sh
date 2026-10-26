#!/bin/bash
# Time-to-HEVC sweep of the rate term in the training loss (rate_weight, compress.rate) on the A100: one phase per variant, 24 tasks each
# (480p and 720p, QP 22-37, budgets 0.5-4 s) with the defaults otherwise (YUV with luma weighting, feature prediction; the sweep arms pin RGB,
# so this sets color_space yuv). Variants:
#   rate-ref           rate_weight 0 (the reference on the same code)
#   rate-w<w>          rate_weight w, chroma bits weighted by uv_weight (rate_uv_weight -1, which keeps the chroma near x265's)
#   rate-w<w>-uv<u>    rate_weight w and rate_uv_weight u (u 1: chroma bits counted fully, which VMAF does not see, at a large colour cost)
# The local 480p sweep (rate-local-*) found w0.25 and w0.5 best with the colour kept; the rate term costs steps (6-8% on the RTX 2050, more
# where a step is shorter), which this sweep measures on the A100.
# Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise; the rate term needs no change of compress/csrc).
# Usage (repository root, on the cluster): slurm/realtime/submit_rate.sh [variant ...]   (default: rate-ref rate-w0.25 rate-w0.5)
# Analysis (locally, with results/realtime/rate-* copied back):
#   .venv/bin/python experiments/realtime/frame_luma.py rate-ref rate-w0.25 rate-w0.5   (one phase at a time on the RTX 2050)
#   .venv/bin/python experiments/init/compare.py rate-ref rate-w0.25 rate-w0.5 --arm qp-lam4-mid-add25-weight \
#       --metrics vmaf,vmaf_neg,psnr_y,psnr_uv --anchor results/lowres/hevc_noinfo.csv
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- rate-ref rate-w0.25 rate-w0.5
fi
common=(--res 480p,720p --arms qp-lam4-mid-add25-weight --rates 22,27,32,37 --budgets 0.5,1,2,4 --set cuda_graph=true --set color_space=yuv)
for variant in "$@"; do
    case "${variant}" in
        rate-ref)
            extra=(--set rate_weight=0) ;;
        rate-w*-uv*)
            weight=${variant#rate-w}
            uv=${weight#*-uv}
            weight=${weight%-uv*}
            extra=(--set "rate_weight=${weight}" --set "rate_uv_weight=${uv}") ;;
        rate-w*)
            extra=(--set "rate_weight=${variant#rate-w}") ;;
        *)
            echo "Unknown variant '${variant}'" >&2
            exit 1 ;;
    esac
    echo "Phase ${variant}:"
    SKIP_SETUP=${SKIP_SETUP:-1} slurm/realtime/submit.sh sweep --phase "${variant}" "${common[@]}" "${extra[@]}"
done
