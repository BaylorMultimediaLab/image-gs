#!/bin/bash
# Time-to-HEVC sweep at 2160p on the A100: the default configuration (YUV with luma weighting, feature prediction, rate term, local-std init,
# geometry coding; the sweep arms pin RGB, so this sets color_space yuv) on frame 1 of the three UVG sequences at 3840x2160, QP 22-37, one
# array task per image and QP (12 tasks), in phase PHASE (default 2160p). BUDGETS (default 0.5,1,2,4 s) sets the budgets: a step at 2160p
# takes about 4 times as long as at 1080p, so 10 and 20 s give about the steps of 2.5 and 5 s there. Extra arguments go to
# slurm/realtime/sweep.py, e.g., --rates 17,22,27,32,37 (with another PHASE, since a sweep stores its rows by task index).
# The frames are media/uvg/2160p/<image>.png, converted from UVG's raw 4:2:0 8-bit sources as the 1080p ones (see Preparation); media/ is not
# in git, so copy that folder to the cluster first. Reuses the environment built by slurm/setup.slurm (SKIP_SETUP=1 unless set otherwise),
# which must include the current compress/csrc (the scale prediction's nearest-neighbor distances). With PARTITION=gpu_a100_debug, the tasks
# run one after the other in one job.
# Usage (repository root, on the cluster): [PHASE=...] [BUDGETS=...] [SKIP_SETUP=0] [PARTITION=...] slurm/realtime/submit_2160p.sh \
#     [sweep.py arguments]
#   e.g., BUDGETS=0.5,1,2,4,10,20 slurm/realtime/submit_2160p.sh
# Preparation (locally, repository root):
#   .venv/bin/python experiments/lowres/extract_frames.py media/uvg/2160p 1 \
#       ~/Downloads/{Beauty,Bosphorus,Jockey}_3840x2160_120fps_420_8bit_YUV_RAW.7z
#   .venv/bin/python experiments/lowres/hevc.py results/lowres/hevc_noinfo_2160p.csv --no-info --res 2160p
# Analysis (locally, with results/realtime/<phase> copied back):
#   .venv/bin/python experiments/realtime/frame_luma.py 2160p
#   .venv/bin/python experiments/init/compare.py 2160p --arm qp-lam4-mid-add25-weight --metrics vmaf,vmaf_neg,psnr_y,psnr_uv,psnr \
#       --anchor results/lowres/hevc_noinfo_2160p.csv
#   .venv/bin/python experiments/realtime/plot.py geo geo-1080p 2160p --budgets 2,4,10,20 --metric vmaf \
#       --anchor results/lowres/hevc_noinfo.csv,results/lowres/hevc_noinfo_1080p.csv,results/lowres/hevc_noinfo_2160p.csv
set -eo pipefail

SKIP_SETUP=${SKIP_SETUP:-1} slurm/realtime/submit.sh sweep --phase "${PHASE:-2160p}" --res 2160p --arms qp-lam4-mid-add25-weight \
    --rates 22,27,32,37 --budgets "${BUDGETS:-0.5,1,2,4}" --set cuda_graph=true --set color_space=yuv "$@"
