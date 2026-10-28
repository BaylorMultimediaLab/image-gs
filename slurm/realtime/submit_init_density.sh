#!/bin/bash
# Time-to-HEVC sweep of the initial positions on the A100: one phase per variant, 24 tasks each (480p and 720p, QP 22-37, budgets 0.5-4 s)
# with the defaults otherwise (YUV with luma weighting, feature prediction, rate term; the sweep arms pin RGB, so this sets color_space yuv).
# Variants:
#   density-ref                    init_mode gradient: fine blobs 0.5, the others by the squared gradient with 30% uniform
#   density-g<gamma>               init_mode local-std: fine blobs 0.5, the others by the local luma standard deviation to the power gamma
#                                  (init_std_gamma), none uniform
#   density-f<fine>-u<uniform>[-g<gamma>]
#                                  init_mode local-std with init_fine_ratio fine and init_random_ratio uniform (the share of the Gaussians
#                                  other than the fine blobs placed uniformly), gamma 0.8 unless given; one decimal for uniform (run names)
# The phase of a variant is its name plus SUFFIX, as a sweep skips the runs its phase already has: a new SUFFIX reruns the reference next to
# the variants (same nodes, same time).
# Results so far: local-std gamma 0.8 with fine 0.5 and no uniform share is better than the default by ~1-2% (VMAF, RGB PSNR), gamma 0.5 not;
# without a uniform share flat areas (the sky of Bosphorus) stay sparse, with coverage holes at step 0. Without arguments, the grid of fine
# 0.3/0.5/0.7 x uniform 0/0.1/0.2/0.3 (gamma 0.8) and the reference: 13 phases.
# Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise; only Python changed). With PARTITION=gpu_a100_debug,
# each variant runs as one job (~5 min each).
# Usage (repository root, on the cluster): [SUFFIX=...] [PARTITION=...] slurm/realtime/submit_init_density.sh [variant ...]
#   e.g., SUFFIX=-grid PARTITION=gpu_a100_debug slurm/realtime/submit_init_density.sh
# Analysis (locally, with results/realtime/density-* copied back; e.g., with SUFFIX=-grid):
#   for p in results/realtime/density-*-grid; do .venv/bin/python experiments/realtime/frame_luma.py $(basename $p); done   (one at a time)
#   .venv/bin/python experiments/init/density_grid.py --suffix=-grid
set -eo pipefail

if [ $# -eq 0 ]; then
    set -- density-ref
    for fine in 0.3 0.5 0.7; do
        for uniform in 0 0.1 0.2 0.3; do
            set -- "$@" "density-f${fine}-u${uniform}"
        done
    done
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
        density-ref)
            extra=(--set init_mode=gradient --set init_random_ratio=0.3) ;;
        density-g*)
            extra=(--set init_mode=local-std --set "init_std_gamma=${variant#density-g}" --set init_random_ratio=0) ;;
        density-f*-u*)
            settings=${variant#density-f}
            fine=${settings%%-*}
            uniform=${settings#*-u}
            uniform=${uniform%%-*}
            gamma=0.8
            if [[ "${settings}" == *-g* ]]; then
                gamma=${settings##*-g}
            fi
            extra=(--set init_mode=local-std --set "init_fine_ratio=${fine}" --set "init_random_ratio=${uniform}" --set "init_std_gamma=${gamma}") ;;
        *)
            echo "Unknown variant '${variant}'" >&2
            exit 1 ;;
    esac
    echo "Phase ${variant}${SUFFIX:-}:"
    SKIP_SETUP=1 AFTER="${after}" slurm/realtime/submit.sh sweep --phase "${variant}${SUFFIX:-}" "${common[@]}" "${extra[@]}"
done
