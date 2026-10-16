"""
Where the time of an optimization step goes in real-time mode, on an otherwise idle GPU: wall-clock time per step (as slurm/lowres/calib.py
measures it), GPU busy time per step (union of all kernels, memory copies, and memsets, timed by the PyTorch profiler), and the CPU side
(kernel launches, CUDA runtime calls, waits for the GPU, time per phase of the step). The GPU idle time per step (wall - busy) bounds what
CUDA graphs can save: a captured step is replayed with one launch, so it takes about its GPU busy time.
Configurations res:N use the settings of the 480p/720p equal-time sweeps (slurm/lowres/sweep.py, arm step-feat64) without pruning.
CPU times under the profiler are inflated by its own overhead (compare profiled_wall_ms with wall_ms), GPU times by a few percent (the busy
time exceeds the unprofiled wall time by up to 7% on a GPU-bound RTX 2050), so gpu_idle_ms slightly underestimates the idle time.
A finished profiler session keeps slowing down kernel launches for the rest of the process (+25% per step on a CPU-bound step), so all
configurations are timed first, and profiled afterwards with new models trained for as many steps.

Usage (repository root): slurm/realtime/step_profile.py --phase <name> [--configs 480p:1000,...] [--warmup 50] [--steps 300] [--profile-steps 50]
                         [--modes eager,graph] [--set key=value ...]  (graph: steps replayed from a CUDA graph, cuda_graph)
Results in results/realtime/<phase>: profile.csv (one row per configuration and mode), profile_kernels.csv (GPU time and launches per kernel
and step), profile_cpu.csv (CPU time and calls per step of the step phases and CUDA runtime calls), and trace_<res>-<N>-<mode>.json.gz (Chrome
trace of the profiled steps, for https://ui.perfetto.dev).
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from time import perf_counter

import torch
import yaml
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "slurm", "lowres"))
import model as model_module  # noqa: E402
from model import GaussianSplatting2D  # noqa: E402
from sweep import ARMS, FIXED, IMAGES  # noqa: E402

CONFIGS = "480p:1000,480p:2000,480p:4000,720p:4500,720p:9000"
ARM = "step-feat64"


def no_range(name):
    return nullcontext()


def step(model, label=no_range):
    """One optimization step as in GaussianSplatting2D.optimize() in real-time mode, without the time and schedule bookkeeping."""
    model.step = getattr(model, "step", 0) + 1  # The step counter of optimize(), which the graph captures log
    if model.cuda_graph:
        with label("graph_step"):
            model._graph_step()
        return
    with label("zero_grad"):
        model.optimizer.zero_grad()
    with label("forward"):
        images, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
    with label("loss"):
        model._get_total_loss(images)
    with label("backward"):
        model.total_loss.backward()
    with label("adam"):
        model.optimizer.step()
    if model.max_scale > 0.0:
        with label("clamp_scale"):
            model._clamp_scale()


@contextmanager
def forward_ranges(model):
    """Profiler ranges around the parts of the forward pass; the rest of it is the image layout copy."""
    targets = [(model, "_get_params", "forward.params"), (model_module, "project_gaussians_2d_scale_rot", "forward.project"),
               (model_module, "rasterize_gaussians_sum", "forward.rasterize")]
    originals = [getattr(obj, name) for obj, name, _ in targets]

    def ranged(function, name):
        def wrapper(*args, **kwargs):
            with record_function(name):
                return function(*args, **kwargs)
        return wrapper

    for (obj, name, label), original in zip(targets, originals):
        setattr(obj, name, ranged(original, label))
    try:
        yield
    finally:
        for (obj, name, _), original in zip(targets, originals):
            setattr(obj, name, original)


def time_steps(model, steps):
    torch.cuda.synchronize()
    begin = perf_counter()
    for _ in range(steps):
        step(model)
    torch.cuda.synchronize()
    return (perf_counter() - begin) / steps


def union_length(intervals):
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def analyze(prof, steps):
    """Per-step summary, GPU time per kernel, and CPU time per phase and CUDA runtime call (times in ms per step)."""
    events = prof.events()
    device = [e for e in events if e.device_type == DeviceType.CUDA and not e.is_user_annotation]
    ranges = [e for e in events if e.device_type == DeviceType.CPU and e.is_user_annotation]
    runtime = [e for e in events if e.device_type == DeviceType.CPU and not e.is_user_annotation and e.name.startswith("cuda")]

    def kind(name):
        return "memcpy" if name.startswith("Memcpy") else "memset" if name.startswith("Memset") else "kernel"

    def per_step(selected):
        table = defaultdict(lambda: [0, 0.0])
        for e in selected:
            table[e.name][0] += 1
            table[e.name][1] += e.time_range.end - e.time_range.start
        return {name: (calls / steps, 1e-3 * us / steps) for name, (calls, us) in table.items()}

    kernels, cpu_ranges, cpu_runtime = per_step(device), per_step(ranges), per_step(runtime)
    summary = {"gpu_busy_ms": 1e-3 * union_length([(e.time_range.start, e.time_range.end) for e in device]) / steps}
    for k in ["kernel", "memcpy", "memset"]:
        summary[f"{k}_ms"] = sum(ms for name, (_, ms) in kernels.items() if kind(name) == k)
        summary[f"{k}s"] = sum(calls for name, (calls, _) in kernels.items() if kind(name) == k)
    launch = [value for name, value in cpu_runtime.items() if name.startswith("cudaLaunchKernel")]
    sync = [value for name, value in cpu_runtime.items() if name in ("cudaStreamSynchronize", "cudaDeviceSynchronize")]
    summary.update({"launch_api_ms": sum(ms for _, ms in launch), "runtime_api_ms": sum(ms for _, ms in cpu_runtime.values()),
                    "syncs": sum(calls for calls, _ in sync), "sync_wait_ms": sum(ms for _, ms in sync)})
    kernel_rows = [{"kind": kind(name), "name": name, "calls": calls, "ms": ms} for name, (calls, ms) in kernels.items()]
    cpu_rows = ([{"kind": "range", "name": name, "calls": calls, "ms": ms} for name, (calls, ms) in cpu_ranges.items()]
                + [{"kind": "runtime", "name": name, "calls": calls, "ms": ms} for name, (calls, ms) in cpu_runtime.items()])
    return summary, sorted(kernel_rows, key=lambda row: -row["ms"]), sorted(cpu_rows, key=lambda row: (row["kind"], -row["ms"]))


def build_model(res, num, image, log_dir, settings):
    with open("cfgs/default.yaml", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    arm_cfg = {key: value[res] if isinstance(value, dict) else value for key, value in ARMS[ARM].items()}  # Per-resolution values
    cfg.update({**FIXED, **arm_cfg, "realtime": True, "num_gaussians": int(num), "prune_ratio": 0.0, "data_root": "media/uvg",
                "input_path": f"{res}/{image}.png", "log_level": "WARNING", "log_dir": log_dir, **settings})
    model = GaussianSplatting2D(argparse.Namespace(**cfg))
    model.worklog.setLevel("WARNING")
    return model


def short(name, width=100):
    name = name.removeprefix("void ")
    return name if len(name) <= width else name[:width - 3] + "..."


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, help="Results go to results/realtime/<phase>")
    parser.add_argument("--configs", default=CONFIGS, help="Configurations res:N (comma-separated)")
    parser.add_argument("--image", default=IMAGES[0], help="Image in media/uvg/<res>/ without extension")
    parser.add_argument("--warmup", type=int, default=50, help="Steps before the measurements")
    parser.add_argument("--steps", type=int, default=300, help="Steps timed without the profiler (wall-clock time per step)")
    parser.add_argument("--profile-steps", type=int, default=50, help="Steps recorded by the profiler, after as many as warmup + steps")
    parser.add_argument("--modes", default="eager", help="Comma-separated: eager (as optimize() without cuda_graph), graph (cuda_graph)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a setting of cfgs/default.yaml (YAML value), e.g., --set ssim_loss_ratio=0")
    options = parser.parse_args()
    options.settings = {key: yaml.safe_load(value) for key, value in (item.split("=", 1) for item in options.set)}
    os.chdir(REPO)
    out_dir = os.path.join(REPO, "results", "realtime", options.phase)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | settings {options.settings}", flush=True)
    rows, kernel_rows, cpu_rows = [], [], []
    with tempfile.TemporaryDirectory() as log_root:
        runs = [(config, mode) for config in options.configs.split(",") for mode in options.modes.split(",")]
        wall_times = {}  # ms per step, before any profiler session
        for config, mode in runs:
            res, num = config.split(":")
            model = build_model(res, num, options.image, os.path.join(log_root, f"{res}-{num}-{mode}"),
                                {**options.settings, "cuda_graph": mode == "graph"})
            time_steps(model, options.warmup)
            wall_times[config, mode] = 1e3 * time_steps(model, options.steps)
            print(f"{res} N{num} {mode}: wall {wall_times[config, mode]:.3f} ms/step", flush=True)
            del model
            torch.cuda.empty_cache()
        for config, mode in runs:
            res, num = config.split(":")
            model = build_model(res, num, options.image, os.path.join(log_root, f"{res}-{num}-{mode}-profiled"),
                                {**options.settings, "cuda_graph": mode == "graph"})
            time_steps(model, options.warmup + options.steps)
            wall_ms = wall_times[config, mode]
            with forward_ranges(model), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                torch.cuda.synchronize()
                begin = perf_counter()
                for _ in range(options.profile_steps):
                    step(model, record_function)
                torch.cuda.synchronize()
                profiled_wall_ms = 1e3 * (perf_counter() - begin) / options.profile_steps
            prof.export_chrome_trace(os.path.join(out_dir, f"trace_{res}-{num}-{mode}.json.gz"))
            with torch.no_grad():
                num_tiles_hit = model_module.project_gaussians_2d_scale_rot(*model._get_params()[:3], model.img_h, model.img_w,
                                                                             model.tile_bounds)[3]
                intersects = int(num_tiles_hit.sum().item())
            summary, kernels, cpu = analyze(prof, options.profile_steps)
            idle_ms = wall_ms - summary["gpu_busy_ms"]
            key = {"res": res, "num_gaussians": int(num), "mode": mode}
            rows.append({**key, "intersects": intersects, "wall_ms": wall_ms, "profiled_wall_ms": profiled_wall_ms, **summary,
                         "gpu_idle_ms": idle_ms, "gpu_idle_pct": 100.0 * idle_ms / wall_ms})
            kernel_rows += [{**key, **row} for row in kernels]
            cpu_rows += [{**key, **row} for row in cpu]

            s = summary
            print(f"\n=== {res} N{num} {mode} ({intersects} tile intersections) ===\n"
                  f"wall {wall_ms:.3f} ms/step (profiled {profiled_wall_ms:.3f}) | GPU busy {s['gpu_busy_ms']:.3f} ms (kernels "
                  f"{s['kernel_ms']:.3f}, memcpy {s['memcpy_ms']:.3f}, memset {s['memset_ms']:.3f}) | GPU idle {idle_ms:.3f} ms "
                  f"({100.0 * idle_ms / wall_ms:.0f}%)\n"
                  f"per step: {s['kernels']:.0f} kernels, {s['memcpys']:.0f} memcpys, {s['memsets']:.0f} memsets, {s['syncs']:.0f} syncs | "
                  f"CPU (profiled): launch API {s['launch_api_ms']:.3f} ms, all CUDA runtime calls {s['runtime_api_ms']:.3f} ms, "
                  f"of which waiting in syncs {s['sync_wait_ms']:.3f} ms", flush=True)
            print("GPU time per step (top 15):")
            for row in kernels[:15]:
                print(f"  {row['ms']:7.4f} ms {100.0 * row['ms'] / s['gpu_busy_ms']:5.1f}% {row['calls']:4.0f}x  {short(row['name'])}")
            print("CPU time per step of the phases (profiled; forward.rasterize includes the wait for the intersection count):")
            for row in cpu:
                if row["kind"] == "range":
                    print(f"  {row['ms']:7.4f} ms  {row['name']}")
            del model, prof
            torch.cuda.empty_cache()
    write_csv(os.path.join(out_dir, "profile.csv"), rows)
    write_csv(os.path.join(out_dir, "profile_kernels.csv"), kernel_rows)
    write_csv(os.path.join(out_dir, "profile_cpu.csv"), cpu_rows)
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
