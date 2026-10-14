"""
Encoding time at which image-gs overtakes HEVC: RD curves of three rate-control arms at a series of time budgets in real-time mode (wall-clock
budgets, persistent process), at 480p/720p on the UVG test frames. Arms (settings loaded from the earlier sweeps):
  fixed-n            fixed numbers of Gaussians, 10% pruned at 1/6 of the budget (slurm/lowres/sweep.py, arm step-feat64)
  qp-lam4            QP rate control, lambda x4 pruning from an overestimate (slurm/qp/sweep.py, arm cap25x4-lam4)
  qp-lam4-mid-add25  the same, starting in the middle and adding Gaussians between the rounds (slurm/qp/sweep.py, cap25x4-lam4-mid-add25)
Pruning rounds and the learning rate decay come at fixed fractions of the budget, i.e., within a few dozen steps at the shortest budgets.
Each task encodes one rate point (number of Gaussians or QP) of one image and arm at every budget, shortest first, in one process after a
discarded warm-up encode, and writes a row per run to results/realtime/<phase>/rows/task<i>.csv; a rerun skips the finished budgets.

Locally:  slurm/realtime/sweep.py --phase <name> [--res ...] [--images ...] [--arms ...] [--budgets ...]   runs all tasks one after another
Slurm:    slurm/realtime/sweep.py --task <i> [...]   runs only task i (one per array task, see slurm/realtime/sweep.slurm)
Count:    slurm/realtime/sweep.py --count [...]      prints the number of tasks (the array size)
Analysis: .venv/bin/python experiments/realtime/analyze.py <phase>
"""
import argparse
import csv
import importlib.util
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOWRES = load_module("lowres_sweep", "slurm/lowres/sweep.py")
QP = load_module("qp_sweep", "slurm/qp/sweep.py")
BUDGETS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# Settings, the option that sets the rate, and its values per resolution
ARMS = {
    "fixed-n": ({**LOWRES.FIXED, **LOWRES.ARMS["step-feat64"]}, "num_gaussians", LOWRES.NUM_GAUSSIANS),
    "qp-lam4": ({**QP.FIXED, **QP.ARMS["cap25x4-lam4"]}, "qp", {res: QP.QPS for res in LOWRES.NUM_GAUSSIANS}),
    "qp-lam4-mid-add25": ({**QP.FIXED, **QP.ARMS["cap25x4-lam4-mid-add25"]}, "qp", {res: QP.QPS for res in LOWRES.NUM_GAUSSIANS}),
}


def run_task(index, job, phase, budgets):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench import encode
    from utils.misc_utils import load_cfg

    res, image, arm, rate = job
    settings, rate_key, _ = ARMS[arm]
    rate = float(rate) if rate_key == "qp" else int(rate)
    out_dir = os.path.join(REPO, "results", "realtime", phase)
    rows_path = os.path.join(out_dir, "rows", f"task{index:03d}.csv")
    os.makedirs(os.path.dirname(rows_path), exist_ok=True)
    rows = list(csv.DictReader(open(rows_path))) if os.path.exists(rows_path) else []
    budgets = [budget for budget in sorted(budgets) if budget not in {float(row["budget"]) for row in rows}]
    if not budgets:
        print(f"Task {index} ({res} {image} {arm} {rate_key} {rate:g}) is already done", flush=True)
        return
    base_args = load_cfg(cfg_path="cfgs/default.yaml", parser=argparse.ArgumentParser()).parse_args([])

    def overrides(budget, log_root, exp_name):
        return {**settings, rate_key: rate, "realtime": True, "max_time": budget, "data_root": os.path.join(REPO, "media", "uvg"),
                "input_path": f"{res}/{image}.png", "log_root": log_root, "exp_name": exp_name}

    with tempfile.TemporaryDirectory() as log_root:  # CUDA initialization and kernel warm-up, which a persistent encoder pays once
        encode(base_args, overrides(budgets[0], log_root, "warm-up"))
    for budget in budgets:
        exp_name = f"{res}/{image}/{arm}/{rate_key}-{rate:g}/time-{budget:g}"
        row = {"res": res, "image": image, "arm": arm, "rate_key": rate_key, "rate": rate, "budget": budget,
               **encode(base_args, overrides(budget, os.path.join(out_dir, "runs"), exp_name))}
        rows.append(row)
        print(" | ".join(f"{key} {value:.3f}" if isinstance(value, float) else f"{key} {value}" for key, value in row.items()), flush=True)
        with open(rows_path, "w", newline="") as file:  # After every run, in case the task dies
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="time-to-hevc", help="Results go to results/realtime/<phase>")
    parser.add_argument("--res", default=",".join(LOWRES.NUM_GAUSSIANS), help="Resolutions (comma-separated)")
    parser.add_argument("--arms", default=",".join(ARMS), help=f"Arms (comma-separated) from: {', '.join(ARMS)}")
    parser.add_argument("--images", default=",".join(LOWRES.IMAGES), help="Images in media/uvg/<res>/ without extension (comma-separated)")
    parser.add_argument("--budgets", default=",".join(f"{budget:g}" for budget in BUDGETS), help="Time budgets in seconds (comma-separated)")
    parser.add_argument("--task", type=int, default=None, help="Run only this task index (Slurm array task)")
    parser.add_argument("--count", action="store_true", help="Print the number of tasks and exit")
    args = parser.parse_args()
    args.res, args.arms, args.images = args.res.split(","), args.arms.split(","), args.images.split(",")
    args.budgets = [float(budget) for budget in args.budgets.split(",")]
    unknown = [arm for arm in args.arms if arm not in ARMS] + [res for res in args.res if res not in LOWRES.NUM_GAUSSIANS]
    if unknown:
        parser.error(f"Unknown arms or resolutions: {', '.join(unknown)}")
    return args


if __name__ == "__main__":
    args = parse_args()
    jobs = [(res, image, arm, rate) for res in args.res for image in args.images for arm in args.arms for rate in ARMS[arm][2][res]]
    if args.count:
        print(len(jobs))
        sys.exit(0)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    import torch
    torch.hub.set_dir("models/torch")
    if args.task is not None:
        if not 0 <= args.task < len(jobs):
            sys.exit(f"Task {args.task} is out of range (there are {len(jobs)} tasks)")
        run_task(args.task, jobs[args.task], args.phase, args.budgets)
    else:
        for index, job in enumerate(jobs):
            run_task(index, job, args.phase, args.budgets)
