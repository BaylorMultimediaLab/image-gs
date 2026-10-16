"""
Wall-clock time per frame of a persistent encoder: one process encodes the frames one after another, constructing a GaussianSplatting2D per
frame, so that imports, CUDA initialization, and kernel warm-up are paid once. Compares the default mode with real-time mode (--realtime)
at a fixed optimization time budget, with the settings of the 480p/720p equal-time sweeps (slurm/lowres/sweep.py).
Usage (repository root): slurm/realtime/bench.py --phase <name> [--budget 1] [--passes 2] [--arms default,realtime] [--res 480p,720p]
Results: results/realtime/<phase>/frames.csv (one row per frame, pass 1 includes the warm-up) and the runs in results/realtime/<phase>/runs.
"""
from time import perf_counter

BEGIN = perf_counter()
import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "slurm", "lowres"))
from main import get_log_dir  # noqa: E402
from model import GaussianSplatting2D  # noqa: E402
from sweep import FIXED, IMAGES  # noqa: E402
from utils.misc_utils import load_cfg  # noqa: E402

IMPORT_TIME = perf_counter() - BEGIN
NUM_GAUSSIANS = {"480p": 2000, "720p": 4500}  # The same number per pixel
ARMS = {"default": {}, "realtime": {"realtime": True}}


def encode(base_args, overrides):
    """
    Encode one frame; returns its timings, steps, numbers of Gaussians, PSNR, SSIM, bitstream size (without quantization, the float32 size of the
    Gaussians), the pruning and densification rounds
    ("P<step>:<number>" or "A<step>:<number>") with their total and slowest wall-clock time, and the CUDA graph captures (cuda_graph).
    """
    args = copy.copy(base_args)
    vars(args).update(overrides)
    args.log_dir = get_log_dir(args)
    torch.cuda.synchronize()
    begin = perf_counter()
    model = GaussianSplatting2D(args)
    torch.cuda.synchronize()
    init_time = perf_counter() - begin
    model.optimize()
    total_time = perf_counter() - begin
    ckpt = torch.load(os.path.join(model.ckpt_dir, f"ckpt_step-{model.step:d}.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(model.log_dir, "log_train.txt"), encoding="utf8") as file:
        rounds = re.findall(r"Step: (\d+) \| (Pruning|Adding) (\d+) Gaussians", file.read())
    return {"num_init": args.num_gaussians, "gaussians": model.num_gaussians, "steps": model.step, "counted_s": model.total_time_accum,
            "init_s": init_time, "optimization_s": model.wall_times["optimization"], "saving_s": model.wall_times["saving"], "total_s": total_time,
            "psnr": ckpt["psnr"], "ssim": ckpt["ssim"], "bytes": ckpt.get("bitstream_bytes", ckpt["bytes"]),
            "rounds": " ".join(f"{action[0]}{step}:{num}" for step, action, num in rounds),
            "round_s": model.round_time_accum, "round_max_s": model.round_time_max[0], "graph_captures": model.graph_captures,
            "graph_capture_s": model.graph_capture_time, "graph_capture_max_s": model.graph_capture_max[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, help="Results go to results/realtime/<phase>")
    parser.add_argument("--budget", type=float, default=1.0, help="Optimization time budget per frame in seconds")
    parser.add_argument("--passes", type=int, default=2, help="Passes over all frames (the first one includes the warm-up)")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"Arms (comma-separated) from: {', '.join(ARMS)}")
    parser.add_argument("--res", default=",".join(NUM_GAUSSIANS), help="Resolutions (comma-separated)")
    parser.add_argument("--images", default=",".join(IMAGES), help="Images in media/uvg/<res>/ without extension (comma-separated)")
    options = parser.parse_args()
    os.chdir(REPO)
    torch.hub.set_dir("models/torch")  # As in main.py (LPIPS weights)
    out_dir = os.path.join(REPO, "results", "realtime", options.phase)
    os.makedirs(out_dir, exist_ok=True)
    base_args = load_cfg(cfg_path="cfgs/default.yaml", parser=argparse.ArgumentParser()).parse_args([])
    print(f"{torch.cuda.get_device_name(0)} | imports {IMPORT_TIME:.2f} s", flush=True)
    rows = []
    for pass_idx in range(1, options.passes + 1):
        for res in options.res.split(","):
            for image in options.images.split(","):
                for arm in options.arms.split(","):
                    overrides = {**FIXED, **ARMS[arm], "num_gaussians": NUM_GAUSSIANS[res], "max_time": options.budget,
                                 "data_root": os.path.join(REPO, "media", "uvg"), "input_path": f"{res}/{image}.png",
                                 "log_root": os.path.join(out_dir, "runs"), "exp_name": f"pass{pass_idx}/{res}/{image}/{arm}"}
                    row = {"pass": pass_idx, "res": res, "image": image, "arm": arm, **encode(base_args, overrides)}
                    rows.append(row)
                    print(" | ".join(f"{key} {value:.3f}" if isinstance(value, float) else f"{key} {value}" for key, value in row.items()),
                          flush=True)
                    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as file:  # After every frame, in case the job dies
                        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    print(f"Results: {os.path.join(out_dir, 'frames.csv')}")


if __name__ == "__main__":
    main()
