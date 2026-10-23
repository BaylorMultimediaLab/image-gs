#!/bin/bash
# Quick time-to-HEVC sweep at 1080p on the A100 debug partition, as one job that runs its tasks one after the other (no array): the default
# configuration (fine-blob init, YUV features with luma weighting, feature prediction; the sweep arms pin RGB, so this sets color_space yuv as
# submit_yuv.sh's yuv-w0.25-r2) on the three UVG frames at their source resolution, QP 22-37, budgets 0.5-4 s: 12 tasks of ~15 s each
# (a warm-up encode and four budgets), a few minutes in all.
# The frames are media/uvg/1080p/<image>.png (frame 1 of the UVG 1920x1080 PNGs, from which media/uvg/480p and 720p are downsampled); media/
# is not in git, so copy that folder to the cluster first.
# Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise). It must include the C++ feature prediction
# (compress/csrc/prediction.cpp), or saving the first bitstream fails; if it was built before that, submit with SKIP_SETUP=0 (the setup job
# then runs first on the same partition).
# Usage (repository root, on the cluster): [PHASE=1080p] [PARTITION=gpu_a100_debug] slurm/realtime/submit_1080p.sh [sweep.py arguments]
#   extra arguments go to slurm/realtime/sweep.py and override the defaults below, e.g., --rates 17,22,27,32,37,42
# Analysis (locally, with results/realtime/<phase> copied back; the anchor is experiments/lowres/hevc.py <csv> --no-info --res 1080p):
#   .venv/bin/python experiments/realtime/frame_luma.py 1080p
#   .venv/bin/python experiments/realtime/plot.py 1080p --anchor results/lowres/hevc_noinfo_1080p.csv --metric vmaf
#   .venv/bin/python experiments/init/compare.py 1080p --arm qp-lam4-mid-add25-weight --metrics psnr,psnr_y,psnr_uv,vmaf,vmaf_neg \
#       --anchor results/lowres/hevc_noinfo_1080p.csv
set -eo pipefail

phase=${PHASE:-1080p}
partition=${PARTITION:-gpu_a100_debug}
mkdir -p results/slurm-logs
dependency=()
if [ "${SKIP_SETUP:-1}" != "1" ]; then
    setup_job=$(sbatch --parsable --partition="${partition}" slurm/setup.slurm)
    dependency=(--dependency="afterok:${setup_job%%;*}")
    echo "Setup job: ${setup_job%%;*}"
fi
sbatch --partition="${partition}" --time=00:20:00 "${dependency[@]}" slurm/realtime/sweep.slurm --phase "${phase}" --res 1080p \
    --arms qp-lam4-mid-add25-weight --rates 22,27,32,37 --budgets 0.5,1,2,4 --set cuda_graph=true --set color_space=yuv "$@"
echo "Sweep submitted as one job (logs in results/slurm-logs, results in results/realtime/${phase})"
