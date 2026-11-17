#!/bin/bash
# Runs the ablation study from Image-GS to the default configuration on this machine's GPU: the first frames of 6 UVG sequences at 4
# resolutions and 4 rate points with a budget of 20 s, for the 7 rows of the paper table and the 2 timing arms
# that rows 0 and 1 need. One encode at a time, so that the budgets and the timings measure this GPU alone.
#   ablation/run.sh                                   # everything
#   ablation/run.sh --res 480p,720p --arms 6-yuv
#   ablation/run.sh --sequences Beauty --res 480p --budget 2 --out /tmp/ablation-test   # smoke test
# This repository runs in the default venv (.venv). The original Image-GS (commit 7f74702 with ablation/imagegs.patch: the input image in
# the log folder) runs in a venv of its own in $IMAGEGS (default ~/.cache/image-gs/imagegs), which this script builds first if it is
# missing: the default venv's packages with Image-GS's gsplat and fused-ssim. Results go to <out>/<arm>/uvg/<res>/<sequence>_0001/<run>
# (default out: results/ablation), the output of each arm's encodes to <out>/<arm>.log. A rerun skips the encodes that completed
# (<out>/<arm>/completed.txt), e.g., after an interruption.
#
# Scoring a run (<run>/checkpoints/ckpt_step-*.pt is its final checkpoint):
#   rows 0-1  image: render_res-*.png (16-bit PNG of the quantized Gaussians, rounded to 8 bits); rate: the checkpoint's "bytes" (Image-GS's
#             16 bits per parameter times the number of Gaussians)
#   row 2     image: the decoded bitstream.rans; rate: fixed-length codes of the quantization indices, not stored (the checkpoint's "bytes"
#             is 0 in real-time mode): the model's _get_fixed_length_bits() / 8 with the final checkpoint loaded. bitstream.rans is the
#             same image coded with rANS (rate: the checkpoint's "bitstream_bytes")
#   rows 3-6  image: the decoded bitstream.rans; rate: the checkpoint's "bitstream_bytes"
#
# Arms, in the order they run:
#   calib-imagegs     timing: the original Image-GS at a fixed number of steps
#   calib-ours        timing: this repository's real-time encoder (our kernels, the real-time loop, and CUDA graphs) in the Image-GS
#                     configuration (the Image-GS settings of all rows below), without writing a bitstream
#   0-imagegs         row 0: the original Image-GS (16 bits) at the steps it makes in the budget (calib-imagegs)
#   1-realtime        row 1: the same at the steps that the real-time encoder makes in the budget (calib-ours)
#   2-quantization    rows 2-6: this repository, the default configuration with the Image-GS settings of all later rows (below), from
#   ... 6-yuv         step quantization (row 2) to the default configuration (row 6)
set -eo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
repo=${PWD}
res=480p,720p,1080p,2160p
sequences=Beauty,Bosphorus,HoneyBee,Jockey,ReadySetGo,ShakeNDry
arms=""
budget=20
out=results/ablation
while [ $# -gt 0 ]; do
    case "$1" in
        --res|--sequences|--arms|--budget|--out) declare "${1#--}=$2"; shift 2 ;;
        --res=*|--sequences=*|--arms=*|--budget=*|--out=*) arg=${1#--}; declare "${arg%%=*}=${arg#*=}"; shift ;;
        *) echo "Unknown argument $1" >&2; exit 1 ;;
    esac
done
out=$(readlink -m "${out}")
imagegs=${IMAGEGS:-${HOME}/.cache/image-gs/imagegs}

qps=(22 27 32 37)
# Per resolution: the numbers of Gaussians that --qp starts from at these QPs (model.py, _init_qp), the rate points without rate control;
# the pruning interval at 20 s (README); Image-GS's position learning rate (5e-4 image widths per step, in pixels); the steps of the
# Image-GS timing runs
declare -A gaussians=([480p]="10199 5209 2660 1359" [720p]="16583 8469 4325 2209" [1080p]="26976 13777 7036 3594"
                      [2160p]="61974 31652 16165 8256")
declare -A interval=([480p]=100 [720p]=50 [1080p]=50 [2160p]=50)
declare -A imagegs_pos_lr=([480p]=0.427 [720p]=0.64 [1080p]=0.96 [2160p]=1.92)
declare -A calib_steps=([480p]=2000 [720p]=1500 [1080p]=800 [2160p]=300)

# Rows 2-6, each with the Image-GS settings that it replaces (POS_LR: imagegs_pos_lr at the resolution). A row runs the default
# configuration with the Image-GS settings of all rows after it, so that row 6 is the default configuration and the settings of all rows
# together are the Image-GS configuration (timing arm calib-ours).
rows=(
    2-quantization    "--qp=0 --qp_step_split=0"  # With --num_gaussians: a fixed number of Gaussians and fixed steps
    3-entropy-coding  "--feat_step_adapt=none --rot_step_adapt=none --pos_step_adapt=none --color_transform=none --no-feat_prediction \
                       --no-scale_prediction"
    4-rd-training     "--qp_lambda_scale=0 --prune_add_ratio=0 --rate_weight=0 --decay_after_steps=0 --pos_lr_px=POS_LR"
    5-init            "--init_scale_knn=0 --init_scale=5 --init_aspect=1 --init_fine_ratio=0 --init_mode=gradient --init_random_ratio=0.3"
    6-yuv             "--color_space=rgb --uv_weight=1 --uv_step_ratio=1"
)
all_arms=(calib-imagegs calib-ours 0-imagegs 1-realtime 2-quantization 3-entropy-coding 4-rd-training 5-init 6-yuv)

row_flags() {  # <row, or "" for all> <resolution>: sets flags to the Image-GS settings of the rows after it
    local found=0 i settings
    if [ -z "$1" ]; then
        found=1
    fi
    flags=()
    for ((i = 0; i < ${#rows[@]}; i += 2)); do
        if [ "${found}" = 1 ]; then
            read -ra settings <<< "${rows[i + 1]}"
            flags+=("${settings[@]//POS_LR/${imagegs_pos_lr[$2]}}")
        fi
        if [ "${rows[i]}" = "$1" ]; then
            found=1
        fi
    done
}

speed() {  # <timing arm> <input> <Gaussians>: the steps per second of optimization of its completed run of the input
    local log
    for log in "${out}/$1/$2/num-$3_"*/log_train.txt; do
        if grep -q "Optimization completed" "${log}" 2>/dev/null; then
            grep -hoE "Wall-clock time: [0-9.]+ s optimization \([0-9]+ steps\)|Step: [0-9]+ \| Total: [0-9.]+ s" "${log}" | tail -1 \
                | awk '{gsub(/\(/, ""); print ($1 == "Step:") ? $2 / $5 : $6 / $3}'
            return
        fi
    done
}

setup_imagegs() {  # Builds the original Image-GS in $imagegs and checks it with a short run
    local build site
    echo "Building the original Image-GS in ${imagegs}"
    rm -rf "${imagegs}"
    mkdir -p "${imagegs}/code"
    if git cat-file -e 7f74702 2>/dev/null; then
        git archive 7f74702 | tar -x -C "${imagegs}/code"
    else
        git clone --quiet https://github.com/NYU-ICL/image-gs.git "${imagegs}/clone"
        git -C "${imagegs}/clone" archive 7f74702 | tar -x -C "${imagegs}/code"
        rm -rf "${imagegs}/clone"
    fi
    patch -d "${imagegs}/code" -p1 < ablation/imagegs.patch
    ln -s "${repo}/models" "${imagegs}/code/models"  # LPIPS weights
    # The default venv's packages after this venv's own, in which Image-GS's gsplat and fused-ssim replace this repository's
    .venv/bin/python -m venv "${imagegs}/venv"
    site=$("${imagegs}/venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    .venv/bin/python -c "import site; print('\n'.join(site.getsitepackages()))" > "${site}/default-venv.pth"
    build=$(mktemp -d)
    cp -r "${imagegs}/code/gsplat" "${build}/gsplat"
    "${imagegs}/venv/bin/python" -m pip install --quiet --no-cache-dir scikit-image==0.26.0
    # Forced: pip would take this repository's gsplat and fused-ssim for them
    MAX_JOBS=$(nproc) "${imagegs}/venv/bin/python" -m pip install --quiet --no-cache-dir --no-deps --no-build-isolation --force-reinstall \
        "${build}/gsplat" "git+https://github.com/rahul-goel/fused-ssim/@77aba120314babc98140cd6c7e8c7119fa92cbee"
    for package in gsplat fused_ssim; do
        "${imagegs}/venv/bin/python" -c "import ${package}, sys; sys.exit(not ${package}.__file__.startswith('${imagegs}/venv/'))"
    done
    rm -rf "${build}"
    (cd "${imagegs}/code" && "${imagegs}/venv/bin/python" main.py --input_path=uvg/480p/Beauty_0001.png --data_root="${repo}/media" \
        --log_root="${imagegs}/smoke" --exp_name=smoke --quantize --max_steps=200 --add_steps=10 --post_min_steps=0 --save_image_format=png)
    grep -q "Optimization completed" "${imagegs}"/smoke/smoke/uvg/480p/Beauty_0001/*/log_train.txt
    touch "${imagegs}/ready"
    echo "Image-GS ready"
}

selected=()
for arm in "${all_arms[@]}"; do
    if [ -z "${arms}" ] || [[ ",${arms}," == *",${arm},"* ]]; then
        selected+=("${arm}")
    fi
done
read -ra resolutions <<< "${res//,/ }"
read -ra inputs <<< "${sequences//,/ }"
for r in "${resolutions[@]}"; do
    for sequence in "${inputs[@]}"; do
        if [ ! -f "media/uvg/${r}/${sequence}_0001.png" ]; then
            echo "Missing input media/uvg/${r}/${sequence}_0001.png" >&2
            exit 1
        fi
    done
done
if [[ " ${selected[*]} " == *imagegs* || " ${selected[*]} " == *" 1-realtime "* ]] && [ ! -f "${imagegs}/ready" ]; then
    setup_imagegs
fi
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)
if [ -n "${busy}" ]; then
    echo "Warning: other processes use the GPU and slow down the timed encodes:"$'\n'"${busy}" >&2
fi

total=$((${#selected[@]} * ${#resolutions[@]} * ${#inputs[@]} * ${#qps[@]}))
count=0
mkdir -p "${out}"
for arm in "${selected[@]}"; do
    mkdir -p "${out}/${arm}"
    for r in "${resolutions[@]}"; do
        read -ra starts <<< "${gaussians[${r}]}"
        for sequence in "${inputs[@]}"; do
            input=uvg/${r}/${sequence}_0001
            for i in "${!qps[@]}"; do
                count=$((count + 1))
                key="${input} QP ${qps[i]} N ${starts[i]}"
                if grep -qxF "${key}" "${out}/${arm}/completed.txt" 2>/dev/null; then
                    continue
                fi
                run=(--input_path="${input}.png" --log_root="${out}" --exp_name="${arm}" --num_gaussians="${starts[i]}")
                if [[ ${arm} == calib-imagegs || ${arm} == 0-imagegs || ${arm} == 1-realtime ]]; then
                    case ${arm} in
                        calib-imagegs) steps=${calib_steps[${r}]} ;;
                        0-imagegs) steps=$(speed calib-imagegs "${input}" "${starts[i]}" || true) ;;
                        1-realtime) steps=$(speed calib-ours "${input}" "${starts[i]}" || true) ;;
                    esac
                    if [ -z "${steps}" ]; then
                        echo "Failed: no completed timing run for ${arm} ${key}" | tee -a "${out}/${arm}.log" >&2
                        continue
                    fi
                    if [ "${arm}" != calib-imagegs ]; then
                        steps=$(awk -v speed="${steps}" -v seconds="${budget}" 'BEGIN {print int(speed * seconds + 0.5)}')
                    fi
                    run+=(--quantize --post_min_steps=0 --save_image_format=png --pos_bits=16 --scale_bits=16 --rot_bits=16 --feat_bits=16
                          --max_steps="${steps}")
                    for schedule in add_steps=500 check_decay_steps=1000 eval_steps=100; do  # Image-GS's schedule of 10000 steps, scaled
                        scaled=$((${schedule#*=} * steps / 10000))
                        run+=(--"${schedule%=*}"=$((scaled > 0 ? scaled : 1)))
                    done
                    command=(env -C "${imagegs}/code" "${imagegs}/venv/bin/python" main.py --data_root="${repo}/media" "${run[@]}")
                else
                    if [ "${arm}" = calib-ours ]; then
                        row_flags "" "${r}"
                        flags+=(--bitstream_codec=none)
                    else
                        row_flags "${arm}" "${r}"
                    fi
                    command=(.venv/bin/python main.py "${run[@]}" --realtime --cuda_graph --prune_max_interval="${interval[${r}]}"
                             --qp="${qps[i]}" "${flags[@]}" --max_time="${budget}")
                fi
                printf "%s [%d/%d] %-21s %s\n" "$(date +%H:%M:%S)" "${count}" "${total}" "${arm}" "${key}"
                echo "${command[*]}" >> "${out}/${arm}.log"
                if "${command[@]}" >> "${out}/${arm}.log" 2>&1; then
                    echo "${key}" >> "${out}/${arm}/completed.txt"
                else
                    echo "Failed: ${arm} ${key} (${out}/${arm}.log)" | tee -a "${out}/${arm}.log" >&2
                fi
                if [[ ${arm} == *imagegs* || ${arm} == 1-realtime ]]; then
                    # Only the final render, checkpoint, and logs are scored: the target, gradient map, and training renders (16-bit PNGs)
                    # would take 70% of the disk space
                    rm -f "${out:?}/${arm}/${input}"/*/{gt,gmap}_res-*.png "${out:?}/${arm}/${input}"/*/train/*.png
                fi
            done
        done
    done
done
echo "Done: results in ${out}"
