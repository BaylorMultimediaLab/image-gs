#!/bin/bash
# Time-to-HEVC sweep of the rate term in the training loss (rate_weight, compress.rate) on the A100: one phase per variant, 24 tasks each
# (480p and 720p, QP 22-37, budgets 0.5-4 s) with the defaults otherwise (YUV with luma weighting, feature prediction; the sweep arms pin RGB,
# so this sets color_space yuv). Variants:
#   rate-ref           rate_weight 0 (the reference on the same code)
#   rate-w<w>          rate_weight w, chroma bits weighted by uv_weight (rate_uv_weight -1, which keeps the chroma near x265's)
#   rate-w<w>-uv<u>    rate_weight w and rate_uv_weight u (u 1: chroma bits counted fully, which VMAF does not see, at a large colour cost)
# The phase of a variant is its name plus SUFFIX (e.g., SUFFIX=-fused gives rate-ref-fused), as a sweep skips the runs its phase already has.
# The local 480p sweep (rate-local-*) found w0.25 and w0.5 best with the colour kept.
# Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise). The rate term runs in fused kernels of gsplat
# (gsplat/gsplat/cuda/csrc/rate.cu), so after they change, submit with SKIP_SETUP=0 (one setup job, which all variants wait for); an
# environment without them fails at the first step.
# With PARTITION=gpu_a100_debug, the setup job and each variant run on the debug partition, one job per variant (~5 min each).
# Usage (repository root, on the cluster): [SUFFIX=...] [PARTITION=...] slurm/realtime/submit_rate.sh [variant ...]
#   default variants: rate-ref, rate-w0.25, rate-w0.5, rate-w1, rate-w2 (the weight in factors of 2), and rate-w0.5-uv0.5 (chroma bits at
#   twice uv_weight)
# Analysis (locally, with results/realtime/rate-* copied back; e.g., with SUFFIX=-fused):
#   for p in ref w0.25 w0.5 w1 w2 w0.5-uv0.5; do .venv/bin/python experiments/realtime/frame_luma.py rate-$p-fused; done   (one at a time)
#   .venv/bin/python experiments/init/compare.py rate-ref-fused rate-w0.25-fused rate-w0.5-fused rate-w1-fused rate-w2-fused \
#       rate-w0.5-uv0.5-fused --arm qp-lam4-mid-add25-weight --metrics vmaf,vmaf_neg,psnr_y,psnr_uv --anchor results/lowres/hevc_noinfo.csv
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- rate-ref rate-w0.25 rate-w0.5 rate-w1 rate-w2 rate-w0.5-uv0.5
fi
after=""
if [ "${SKIP_SETUP:-1}" != "1" ]; then
    mkdir -p results/slurm-logs
    setup_job=$(sbatch --parsable ${PARTITION:+--partition="${PARTITION}"} slurm/setup.slurm)
    after=${setup_job%%;*}
    echo "Setup job: ${after}"
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
    echo "Phase ${variant}${SUFFIX:-}:"
    SKIP_SETUP=1 AFTER="${after}" slurm/realtime/submit.sh sweep --phase "${variant}${SUFFIX:-}" "${common[@]}" "${extra[@]}"
done
