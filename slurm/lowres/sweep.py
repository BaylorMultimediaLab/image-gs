"""
Step-size and quantization comparison at 480p/720p at equal encoding time (time-budgeted optimization, max_time).
Runs are numbered in a fixed order (resolution, image, number of Gaussians, arm). Finished runs with the expected settings are skipped.
Run from the repository root; results go to results/lowres/<phase>.

Locally:  slurm/lowres/sweep.py [--res 480p,720p] [--arms ...]   runs all runs one at a time, so that time budgets are not shared
          (--snapshot DIR runs a frozen copy of the code and config, so that repo edits cannot leak into a long sweep)
Slurm:    slurm/lowres/sweep.py --task <i> [...]                 runs only run i (one per array task, see slurm/lowres/sweep.slurm)
Count:    slurm/lowres/sweep.py --count [...]                    prints the number of runs (the array size)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep_utils import IMAGES, REPO, TIME_BUDGET, run_main  # noqa: E402

# Gaussians before pruning (10% are pruned); 720p uses 2.25x as many as 480p, the same number per pixel
NUM_GAUSSIANS = {"480p": [500, 1000, 2000, 4000], "720p": [1125, 2250, 4500, 9000]}
# Time-budgeted optimization (TIME_BUDGET): no progressive optimization, prune 10% at 1/6 and decay the LR at 5/6 of the budget.
# max_steps only caps runaway runs.
FIXED = {"seed": 123, "max_steps": 1000000, "disable_prog_optim": True, "prune_ratio": 0.1, "prune_score": "loo", "prune_time_ratio": 0.1667,
         "decay_time_ratio": 0.8333, "color_space": "rgb", "init_scale": 2.25, "bitstream_codec": "rans", "color_transform": "subtract-green",
         "save_image_format": "png"}
STEP = {"quant_mode": "step", "pos_step": 0.125, "scale_step": 0.03125, "rot_step": 0.703125, "feat_step": 0.03125}
ARMS = {
    "minmax-16-8-8-10": {"quant_mode": "minmax", "pos_bits": 16, "scale_bits": 8, "rot_bits": 8, "feat_bits": 10},
    "step": STEP,
    "step-feat64": {**STEP, "feat_step": 0.015625},
    "step-pos16": {**STEP, "pos_step": 0.0625},
    "step-pos16-feat64": {**STEP, "pos_step": 0.0625, "feat_step": 0.015625},
    "step-feat128": {**STEP, "feat_step": 0.0078125},
    # Step quantization keeps a few very large Gaussians (up to ~350 px), which slow down rasterization: caps of 1/30, 1/15, and 2/15 of
    # the height. They did not pay off at 480p (2026-10-12), so they are not in the default arms.
    "step-maxscale16": {**STEP, "max_scale": {"480p": 16.0, "720p": 24.0}},
    "step-maxscale32": {**STEP, "max_scale": {"480p": 32.0, "720p": 48.0}},
    "step-maxscale64": {**STEP, "max_scale": {"480p": 64.0, "720p": 96.0}},
}
DEFAULT_ARMS = ["minmax-16-8-8-10", "step", "step-feat64", "step-feat128", "step-pos16", "step-pos16-feat64"]


def run(job, phase, budgets, cwd):
    res, image, num_gaussians, arm = job
    exp = f"lowres/{phase}/{res}/{image}/{arm}/N{num_gaussians}"
    arm_cfg = {key: value[res] if isinstance(value, dict) else value for key, value in ARMS[arm].items()}  # Per-resolution values
    expected = {**FIXED, **arm_cfg, "num_gaussians": num_gaussians, "max_time": budgets[res]}
    return run_main(exp, f"{res}/{image}.png", expected, cwd, f"{res} {image:15s} {arm:18s} N{num_gaussians:<5d}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", default=",".join(NUM_GAUSSIANS), help="Resolutions (comma-separated)")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"Arms (comma-separated) from: {', '.join(ARMS)}")
    parser.add_argument("--images", default=",".join(IMAGES), help="Images in media/uvg/<res>/ without extension (comma-separated)")
    parser.add_argument("--phase", default="equal-time", help="Results go to results/lowres/<phase>")
    parser.add_argument("--budget", default=",".join(f"{res}={seconds:g}" for res, seconds in TIME_BUDGET.items()),
                        help="Optimization time budget in seconds per resolution, e.g. 480p=40,720p=90")
    parser.add_argument("--task", type=int, default=None, help="Run only this run index (Slurm array task)")
    parser.add_argument("--count", action="store_true", help="Print the number of runs and exit")
    parser.add_argument("--snapshot", default=None, help="Run a frozen copy of the code and config in this directory instead of the repository")
    args = parser.parse_args()
    args.res, args.arms, args.images = args.res.split(","), args.arms.split(","), args.images.split(",")
    args.budget = {res: float(seconds) for res, seconds in (item.split("=") for item in args.budget.split(","))}
    unknown = [arm for arm in args.arms if arm not in ARMS] + [res for res in args.res if res not in NUM_GAUSSIANS or res not in args.budget]
    if unknown:
        parser.error(f"Unknown arms or resolutions without a budget: {', '.join(unknown)}")
    return args


if __name__ == "__main__":
    args = parse_args()
    # Image-major order, so that complete RD curves of all arms finish early
    jobs = [(res, image, n, arm) for res in args.res for image in args.images for n in NUM_GAUSSIANS[res] for arm in args.arms]
    if args.count:
        print(len(jobs))
        sys.exit(0)
    cwd = REPO if args.snapshot is None else os.path.abspath(args.snapshot)
    if args.task is not None:
        if not 0 <= args.task < len(jobs):
            sys.exit(f"Task {args.task} is out of range (there are {len(jobs)} runs)")
        sys.exit(0 if run(jobs[args.task], args.phase, args.budget, cwd) else 1)
    for job in jobs:
        run(job, args.phase, args.budget, cwd)
    print("DONE", flush=True)
