"""
Helpers shared by the sweeps in slurm/<study>/sweep.py: paths, the UVG test frames and time budgets of the 480p/720p equal-time studies, and
running main.py unless a finished run with the same settings exists.
"""
import glob
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.path.join(REPO, "media", "uvg")
IMAGES = ["Beauty_0001", "Bosphorus_0001", "Jockey_0001"]
# Equal encoding time (user, 2026-10-12): the same optimization time budget for every run of a resolution, measured with GPU synchronization
TIME_BUDGET = {"480p": 40.0, "720p": 90.0}


def finished_ok(exp_dir, expected):
    import yaml  # Not needed to count runs, e.g., on a login node
    for cfg_path in glob.glob(f"{exp_dir}/*/cfg_train.yaml"):
        with open(cfg_path) as f:
            cfg = yaml.safe_load("".join(line for line in f if not line.startswith("#")))
        log = open(os.path.join(os.path.dirname(cfg_path), "log_train.txt")).read()
        if "LPIPS:" in log and "Lossless decoding verified" in log and all(cfg.get(k) == v for k, v in expected.items()):
            return True
    return False


def run_main(exp, input_path, expected, cwd, label):
    """
    Run main.py on media/uvg/<input_path> with the expected settings into results/<exp>, unless a finished run with these settings exists there.
    Returns whether a finished run exists afterwards.
    """
    if finished_ok(f"{REPO}/results/{exp}", expected):
        print(f"{label} skipped (already done)", flush=True)
        return True
    cmd = [sys.executable, "main.py", f"--log_root={REPO}/results", f"--data_root={DATA_ROOT}", f"--input_path={input_path}",
           f"--exp_name={exp}"]
    cmd += [f"--{k}={v}" for k, v in expected.items() if not isinstance(v, bool)] + [f"--{k}" for k, v in expected.items() if v is True]
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    ok = finished_ok(f"{REPO}/results/{exp}", expected)
    print(f"{label} {'done' if ok else 'FAILED'}", flush=True)
    if not ok:
        print((out.stdout + out.stderr)[-4000:], flush=True)
    return ok
