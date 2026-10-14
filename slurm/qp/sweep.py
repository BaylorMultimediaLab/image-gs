"""
QP rate control at 480p/720p at equal encoding time: each QP sets lambda, the feature step, and a content-agnostic overestimate of the number of
Gaussians, which lambda pruning reduces per image (model.py, _init_qp). The arms vary the pruning rounds and the overestimate.
Same images, budgets, and schedule as slurm/lowres/sweep.py, so the fixed-N runs of its equal-time phase are the reference
(experiments/qp/analyze.py). Runs are numbered in a fixed order (resolution, image, QP, arm). Finished runs with the expected settings are skipped.
Run from the repository root; results go to results/qp/<phase>.

Locally:  slurm/qp/sweep.py [--res 480p,720p] [--qps 27,32,37,42] [--arms ...]   runs all runs one at a time
Slurm:    slurm/qp/sweep.py --task <i> [...]                                       runs only run i (see slurm/qp/sweep.slurm)
Count:    slurm/qp/sweep.py --count [...]                                          prints the number of runs (the array size)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sweep_utils import IMAGES, REPO, TIME_BUDGET, run_main  # noqa: E402

QPS = [27, 32, 37, 42]
# Pruning rounds at 1/6, 2/6, ... of the budget (prune_time_ratio) and a single LR decay at 5/6, as in the lowres sweep. The geometry steps
# are pinned; the feature step comes from the QP.
FIXED = {"seed": 123, "max_steps": 1000000, "disable_prog_optim": True, "prune_score": "loo", "prune_time_ratio": 0.1667,
         "decay_time_ratio": 0.8333, "color_space": "rgb", "init_scale": 2.25, "bitstream_codec": "rans", "color_transform": "subtract-green",
         "save_image_format": "png", "quant_mode": "step", "pos_step": 0.125, "scale_step": 0.03125, "rot_step": 0.703125,
         "qp_overestimate": 1.25}
ARMS = {
    "cap10x4": {"prune_max_ratio": 0.1, "prune_times": 4},
    "cap25x4": {"prune_max_ratio": 0.25, "prune_times": 4},
    "cap50x4": {"prune_max_ratio": 0.5, "prune_times": 4},
    "cap25x4-over1": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_overestimate": 1.0},
    "cap25x4-over2": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_overestimate": 2.0},
    "cap25x4-lam2": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 2.0},
    "cap25x4-lam4": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 4.0},
    "cap25x4-over2-lam2": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_overestimate": 2.0, "qp_lambda_scale": 2.0},
    "cap25x4-over2-lam4": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_overestimate": 2.0, "qp_lambda_scale": 4.0},
    # 4 rounds of at most 25% leave at least 32% of the Gaussians, which lambda x4 almost reaches on easy content
    "cap50x4-lam8": {"prune_max_ratio": 0.5, "prune_times": 4, "qp_lambda_scale": 8.0},
    # Start lower and add Gaussians at high-error pixels after each round but the last, which the next round keeps only if they pay off:
    # from the middle (0.6x: between the lambda x4 results of easy and dense content) or below what any content needs (0.4x)
    "cap25x4-lam4-mid-add25": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 4.0, "qp_overestimate": 0.6, "prune_add_ratio": 0.25},
    "cap25x4-lam4-low-add50": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 4.0, "qp_overestimate": 0.4, "prune_add_ratio": 0.5},
    "cap25x4-lam2-mid-add25": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 2.0, "qp_overestimate": 0.6, "prune_add_ratio": 0.25},
    "cap25x4-lam8-mid-add25": {"prune_max_ratio": 0.25, "prune_times": 4, "qp_lambda_scale": 8.0, "qp_overestimate": 0.6, "prune_add_ratio": 0.25},
}
# Local test at 480p (RTX 2050, Beauty + Jockey, 2026-10-13), BD-rate vs fixed N / vs x265 (PSNR): overestimate and prune with lambda x1
# -12% / +2%, x4 -20% / -9% (the cap does not matter, larger overestimates help less); starting in the middle and adding between the rounds
# -27% / -19% (from below: -26% / -17%); growing to the same number of Gaussians as lambda x4 without pruning -9% / +1%
DEFAULT_ARMS = ["cap25x4-lam4", "cap25x4-lam4-mid-add25", "cap25x4-lam4-low-add50", "cap25x4-lam2-mid-add25", "cap25x4-lam8-mid-add25"]


def run(job, phase, budgets, cwd):
    res, image, qp, arm = job
    exp = f"qp/{phase}/{res}/{image}/{arm}/QP{qp:g}"
    expected = {**FIXED, **ARMS[arm], "qp": float(qp), "max_time": budgets[res]}
    return run_main(exp, f"{res}/{image}.png", expected, cwd, f"{res} {image:15s} {arm:18s} QP{qp:<3g}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--res", default=",".join(TIME_BUDGET), help="Resolutions (comma-separated)")
    parser.add_argument("--qps", default=",".join(str(qp) for qp in QPS), help="Quantization parameters (comma-separated)")
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help=f"Arms (comma-separated) from: {', '.join(ARMS)}")
    parser.add_argument("--images", default=",".join(IMAGES), help="Images in media/uvg/<res>/ without extension (comma-separated)")
    parser.add_argument("--phase", default="equal-time", help="Results go to results/qp/<phase>")
    parser.add_argument("--budget", default=",".join(f"{res}={seconds:g}" for res, seconds in TIME_BUDGET.items()),
                        help="Optimization time budget in seconds per resolution, e.g. 480p=40,720p=90")
    parser.add_argument("--task", type=int, default=None, help="Run only this run index (Slurm array task)")
    parser.add_argument("--count", action="store_true", help="Print the number of runs and exit")
    parser.add_argument("--snapshot", default=None, help="Run a frozen copy of the code and config in this directory instead of the repository")
    args = parser.parse_args()
    args.res, args.arms, args.images = args.res.split(","), args.arms.split(","), args.images.split(",")
    args.qps = [float(qp) for qp in args.qps.split(",")]
    args.budget = {res: float(seconds) for res, seconds in (item.split("=") for item in args.budget.split(","))}
    unknown = [arm for arm in args.arms if arm not in ARMS] + [res for res in args.res if res not in args.budget]
    if unknown:
        parser.error(f"Unknown arms or resolutions without a budget: {', '.join(unknown)}")
    return args


if __name__ == "__main__":
    args = parse_args()
    # Image-major order, so that complete RD curves of all arms finish early
    jobs = [(res, image, qp, arm) for res in args.res for image in args.images for qp in args.qps for arm in args.arms]
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
