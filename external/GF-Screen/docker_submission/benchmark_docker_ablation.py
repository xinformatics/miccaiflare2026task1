#!/usr/bin/env python3
"""Benchmark Docker wall time and GPU memory-time under evaluation.txt limits."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path


def gpu_used_mib(index: int) -> float:
    result = subprocess.run(
        [
            "nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="mayo_clinic_ablation:latest")
    parser.add_argument("--mode", choices=("gf", "anchor", "proxy", "proxy_stu"), required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--container-command", default="sh predict.sh")
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--sample-seconds", default=0.10, type=float)
    parser.add_argument("--focus-batch-size", default=1, choices=(1, 2, 4, 8), type=int)
    parser.add_argument("--glance-batch-size", default=4, choices=(1, 2, 4, 8, 16, 32), type=int)
    parser.add_argument("--biomed-batch-size", default=4, choices=(2, 4, 8), type=int)
    parser.add_argument("--crop-overlap", default=0.75, choices=(0.50, 0.625, 0.75), type=float)
    parser.add_argument("--adaptive-selected-threshold", default=0, type=int)
    parser.add_argument(
        "--manual-gpu-passthrough",
        action="store_true",
        help="pass NVIDIA devices/libraries explicitly when the local Docker NVIDIA runtime is absent",
    )
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = gpu_used_mib(args.gpu)
    samples: list[tuple[float, float]] = []
    stopped = threading.Event()

    def sample() -> None:
        origin = time.perf_counter()
        while not stopped.is_set():
            try:
                samples.append((time.perf_counter() - origin, max(0.0, gpu_used_mib(args.gpu) - baseline)))
            except subprocess.SubprocessError:
                pass
            stopped.wait(args.sample_seconds)

    gpu_arguments = ["--gpus", f"device={args.gpu}"]
    if args.manual_gpu_passthrough:
        gpu_arguments = [
            "--device", "/dev/nvidiactl",
            "--device", f"/dev/nvidia{args.gpu}",
            "--device", "/dev/nvidia-uvm",
            "--device", "/dev/nvidia-uvm-tools",
            "-v", "/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro",
            "-v", "/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro",
        ]
    command = [
        "docker", "container", "run", *gpu_arguments, "-m", "28G",
        "--rm", "--network", "none", "-e", f"MAYO_ABLATION={args.mode}",
        "-e", f"MAYO_FOCUS_BATCH_SIZE={args.focus_batch_size}",
        "-e", f"MAYO_GLANCE_BATCH_SIZE={args.glance_batch_size}",
        "-e", f"MAYO_BIOMED_BATCH_SIZE={args.biomed_batch_size}",
        "-e", f"MAYO_CROP_OVERLAP={args.crop_overlap}",
        "-e", f"MAYO_ADAPTIVE_SELECTED_THRESHOLD={args.adaptive_selected_threshold}",
        "-v", f"{args.input.resolve()}:/workspace/inputs:ro",
        "-v", f"{args.output.resolve()}:/workspace/outputs",
    ]
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        command += ["-v", f"{args.tmp_dir.resolve()}:/tmp"]
    command += [args.image, "/bin/bash", "-c", args.container_command]
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    completed = subprocess.run(command)
    wall = time.perf_counter() - started
    stopped.set()
    thread.join()
    area = 0.0
    excess_area = 0.0
    for (left_t, left_m), (right_t, right_m) in zip(samples, samples[1:]):
        duration = right_t - left_t
        area += duration * (left_m + right_m) / 2.0
        excess_area += duration * (max(0.0, left_m - 4096.0) + max(0.0, right_m - 4096.0)) / 2.0
    payload = {
        "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
        "mode": args.mode,
        "focus_batch_size": args.focus_batch_size,
        "glance_batch_size": args.glance_batch_size,
        "biomed_batch_size": args.biomed_batch_size,
        "crop_overlap": args.crop_overlap,
        "adaptive_selected_threshold": args.adaptive_selected_threshold,
        "command": command,
        "exit_code": completed.returncode,
        "wall_seconds_including_container_startup": wall,
        "hard_60s_eligible": completed.returncode == 0 and wall <= 60.0,
        "operating_50s_eligible": completed.returncode == 0 and wall <= 50.0,
        "gpu_index": args.gpu,
        "baseline_gpu_memory_mib": baseline,
        "peak_incremental_gpu_memory_mib": max((value for _, value in samples), default=0.0),
        "gpu_memory_time_mib_seconds": area,
        "gpu_memory_time_above_4096_mib_seconds": excess_area,
        "sample_interval_seconds": args.sample_seconds,
        "sample_count": len(samples),
        "output_count": len(list(args.output.glob("*.nii.gz"))),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
