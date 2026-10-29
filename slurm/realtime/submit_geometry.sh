#!/bin/bash
# Time-to-HEVC sweep of the geometry coding options on the A100: per-Gaussian rotation steps (rot_step_adapt aspect), per-Gaussian position
# steps with two-stage positions (pos_step_adapt major), and the scale prediction (scale_prediction), which all change only the saved model and
# its bitstream. One phase per variant, QP 22-37 on the three UVG frames, with the defaults otherwise (YUV with luma weighting, feature
# prediction, rate term, local-std init; the sweep arms pin RGB, so this sets color_space yuv). Variants:
#   geo-ref    the three options off (the reference on the same code)
#   geo        all three on
#   geo-rot, geo-pos, geo-scale
#              one each; not in the defaults, as a paired re-encode of geo-ref gives them locally (see below)
# RES (default 480p,720p) and BUDGETS (default 0.5,1,2,4 s) set the grid. 1080p goes to phases of its own (<variant>-1080p), since a sweep
# stores its rows by task index; it needs media/uvg/1080p on the cluster (not in git, see submit_1080p.sh). More budgets can be added to a
# finished phase later with the same RES (a sweep skips the budgets that its rows already have), e.g., BUDGETS=10,20. The phase is the variant
# (and -1080p) plus SUFFIX. A task encodes all its budgets in one process: ~1.5 min with budgets up to 20 s, within the task time limit.
# compress/csrc changed (the scale prediction's nearest-neighbor distances), so by default the environment is rebuilt first
# (slurm/setup.slurm, which all phases wait for); SKIP_SETUP=1 reuses an environment built from this code. With PARTITION=gpu_a100_debug,
# the setup job and each phase run on the debug partition, one job per phase.
# Usage (repository root, on the cluster): [RES=...] [BUDGETS=...] [SUFFIX=...] [SKIP_SETUP=1] [PARTITION=...] \
#     slurm/realtime/submit_geometry.sh [variant ...]
#   e.g., RES=480p,720p,1080p BUDGETS=0.5,1,2,4,10,20 slurm/realtime/submit_geometry.sh   (geo-ref, geo, geo-ref-1080p, geo-1080p: 72 tasks)
# Analysis (locally, with results/realtime/geo-* copied back):
#   for p in geo-ref geo geo-ref-1080p geo-1080p; do .venv/bin/python experiments/realtime/frame_luma.py $p; done   (one at a time)
#   .venv/bin/python experiments/init/compare.py geo-ref geo --arm qp-lam4-mid-add25-weight --metrics vmaf,vmaf_neg,psnr_y,psnr_uv,psnr \
#       --anchor results/lowres/hevc_noinfo.csv   (1080p: geo-ref-1080p geo-1080p with --anchor results/lowres/hevc_noinfo_1080p.csv)
#   Paired, on the same trained Gaussians (no run-to-run noise), e.g., for all three options:
#   .venv/bin/python experiments/size-adaptive/reencode_phase.py geo-ref recode --set rot_step_adapt=aspect --set pos_step_adapt=major \
#       --set scale_prediction=true, then frame_luma.py geo-ref-recode and compare.py geo-ref geo-ref-recode (same arguments as above)
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- geo-ref geo
fi
res=${RES:-480p,720p}
budgets=${BUDGETS:-0.5,1,2,4}
low_res=$(echo "${res}" | tr ',' '\n' | grep -vx 1080p | paste -sd, - || true)
high_res=$(echo "${res}" | tr ',' '\n' | grep -x 1080p || true)
after=""
if [ "${SKIP_SETUP:-0}" != "1" ]; then
    mkdir -p results/slurm-logs
    setup_job=$(sbatch --parsable ${PARTITION:+--partition="${PARTITION}"} slurm/setup.slurm)
    after=${setup_job%%;*}
    echo "Setup job: ${after}"
fi
common=(--arms qp-lam4-mid-add25-weight --rates 22,27,32,37 --budgets "${budgets}" --set cuda_graph=true --set color_space=yuv)
for variant in "$@"; do
    rot=none
    pos=none
    scale=false
    case "${variant}" in
        geo-ref) ;;
        geo)
            rot=aspect
            pos=major
            scale=true ;;
        geo-rot)
            rot=aspect ;;
        geo-pos)
            pos=major ;;
        geo-scale)
            scale=true ;;
        *)
            echo "Unknown variant '${variant}'" >&2
            exit 1 ;;
    esac
    extra=(--set "rot_step_adapt=${rot}" --set "pos_step_adapt=${pos}" --set "scale_prediction=${scale}")
    if [ -n "${low_res}" ]; then
        echo "Phase ${variant}${SUFFIX:-} (${low_res}):"
        SKIP_SETUP=1 AFTER="${after}" slurm/realtime/submit.sh sweep --phase "${variant}${SUFFIX:-}" --res "${low_res}" "${common[@]}" "${extra[@]}"
    fi
    if [ -n "${high_res}" ]; then
        echo "Phase ${variant}-1080p${SUFFIX:-} (1080p):"
        SKIP_SETUP=1 AFTER="${after}" slurm/realtime/submit.sh sweep --phase "${variant}-1080p${SUFFIX:-}" --res 1080p "${common[@]}" "${extra[@]}"
    fi
done
