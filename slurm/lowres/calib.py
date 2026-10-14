"""
Time per optimization step (forward, loss, backward, Adam) on an otherwise idle GPU, without evaluation or logging.
Usage (repo root): slurm/lowres/calib.py <out_csv> <res:N:arm,...>  (arms as in slurm/lowres/sweep.py)
"""
import argparse
import csv
import os
import sys
import tempfile
from time import perf_counter

import torch
import yaml

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GaussianSplatting2D  # noqa: E402
from sweep import ARMS, FIXED, IMAGES  # noqa: E402

WARMUP, STEPS = 50, 300


def time_steps(model, steps):
    torch.cuda.synchronize()
    begin = perf_counter()
    for _ in range(steps):
        model.optimizer.zero_grad()
        images, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
        model._get_total_loss(images)
        model.total_loss.backward()
        model.optimizer.step()
    torch.cuda.synchronize()
    return (perf_counter() - begin) / steps


def time_params(model, steps):
    torch.cuda.synchronize()
    begin = perf_counter()
    for _ in range(steps):
        model._get_params()
    torch.cuda.synchronize()
    return (perf_counter() - begin) / steps


rows = []
with tempfile.TemporaryDirectory() as log_root:
    for job in sys.argv[2].split(","):
        res, num, arm = job.split(":")
        with open("cfgs/default.yaml", encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
        arm_cfg = {key: value[res] if isinstance(value, dict) else value for key, value in ARMS[arm].items()}  # Per-resolution values
        cfg.update({**FIXED, **arm_cfg, "num_gaussians": int(num), "prune_ratio": 0.0, "data_root": "media/uvg",
                    "input_path": f"{res}/{IMAGES[0]}.png", "log_level": "WARNING"})
        cfg["log_dir"] = os.path.join(log_root, f"{res}-{num}-{arm}")
        model = GaussianSplatting2D(argparse.Namespace(**cfg))
        model.worklog.setLevel("WARNING")
        time_steps(model, WARMUP)
        step_time = time_steps(model, STEPS)
        with torch.no_grad():
            params_time = time_params(model, STEPS)
        rows.append({"res": res, "num_gaussians": int(num), "arm": arm, "step_ms": 1e3 * step_time, "params_ms": 1e3 * params_time})
        print(f"{res} N{num:>5s} {arm:18s} step {1e3*step_time:6.2f} ms | quantized params (no grad) {1e3*params_time:5.2f} ms", flush=True)
        del model
        torch.cuda.empty_cache()
with open(sys.argv[1], "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
