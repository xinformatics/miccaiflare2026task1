#!/usr/bin/env python3
"""Create a reproducible runtime/storage preflight for the long Focus run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
from pathlib import Path


TRAIN_RE = re.compile(r"Final training\s+\d+/\d+.*?time\s+([0-9.]+)s")
VAL_RE = re.compile(r"Final validation\s+\d+/\d+.*?time\s+([0-9.]+)s")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_log(path: Path, samples_per_epoch: int) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    training = [float(value) for value in TRAIN_RE.findall(text)]
    validation = [float(value) for value in VAL_RE.findall(text)]
    if not training:
        raise ValueError(f"no completed training epochs in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "samples_per_epoch": samples_per_epoch,
        "completed_training_epochs": len(training),
        "training_seconds": {
            "median": statistics.median(training),
            "p25": quantile(training, 0.25),
            "p75": quantile(training, 0.75),
            "min": min(training),
            "max": max(training),
        },
        "completed_validations": len(validation),
        "validation_seconds": ({
            "median": statistics.median(validation),
            "p25": quantile(validation, 0.25),
            "p75": quantile(validation, 0.75),
            "min": min(validation),
            "max": max(validation),
        } if validation else None),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--reference-samples", type=int, required=True)
    parser.add_argument("--target-samples", type=int, default=8192)
    parser.add_argument("--target-epochs", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--checkpoint-example", type=Path, required=True)
    parser.add_argument("--worst-case-checkpoint", type=Path)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.reference_samples, args.target_samples, args.target_epochs, args.val_every) <= 0:
        raise ValueError("sample, epoch, and validation counts must be positive")
    for path in (args.reference_log, args.checkpoint_example, args.storage_root):
        if not path.exists():
            raise FileNotFoundError(path)

    reference = parse_log(args.reference_log, args.reference_samples)
    scale = args.target_samples / args.reference_samples
    validations = math.ceil(args.target_epochs / args.val_every)
    val_stats = reference["validation_seconds"]
    val_median = val_stats["median"] if val_stats else 0.0
    train_stats = reference["training_seconds"]

    def projected_hours(train_seconds: float, validation_seconds: float) -> float:
        return (train_seconds * scale * args.target_epochs + validation_seconds * validations) / 3600.0

    checkpoint_bytes = args.checkpoint_example.stat().st_size
    worst_path = args.worst_case_checkpoint
    worst_bytes = worst_path.stat().st_size if worst_path else checkpoint_bytes
    # One checkpoint per epoch, plus one atomic temporary and one final-model copy.
    retained_files = args.target_epochs + 2
    expected_peak_bytes = checkpoint_bytes * retained_files
    conservative_peak_bytes = worst_bytes * retained_files
    disk = shutil.disk_usage(args.storage_root)
    storage_pass = disk.free >= conservative_peak_bytes * 2

    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "PASS" if storage_pass else "FAIL",
        "purpose": "preflight only; public and hidden predictions are not read",
        "reference": reference,
        "projection": {
            "target_samples_per_epoch_global": args.target_samples,
            "target_epochs": args.target_epochs,
            "validation_every_epochs": args.val_every,
            "validation_events": validations,
            "linear_sample_scale": scale,
            "training_seconds_per_epoch_median": train_stats["median"] * scale,
            "training_seconds_per_epoch_p25": train_stats["p25"] * scale,
            "training_seconds_per_epoch_p75": train_stats["p75"] * scale,
            "total_hours_median": projected_hours(train_stats["median"], val_median),
            "total_hours_interquartile": [
                projected_hours(train_stats["p25"], val_stats["p25"] if val_stats else 0.0),
                projected_hours(train_stats["p75"], val_stats["p75"] if val_stats else 0.0),
            ],
            "caveat": "first-epoch cache construction and concurrent I/O can increase elapsed time",
        },
        "storage": {
            "root": str(args.storage_root.resolve()),
            "free_bytes": disk.free,
            "reference_checkpoint": {
                "path": str(args.checkpoint_example.resolve()),
                "bytes": checkpoint_bytes,
            },
            "worst_case_checkpoint": {
                "path": str(worst_path.resolve()) if worst_path else str(args.checkpoint_example.resolve()),
                "bytes": worst_bytes,
            },
            "retained_file_allowance": retained_files,
            "expected_peak_bytes": expected_peak_bytes,
            "conservative_peak_bytes": conservative_peak_bytes,
            "required_safety_factor": 2.0,
            "headroom_over_conservative_peak": disk.free / conservative_peak_bytes,
            "decision": "PASS" if storage_pass else "FAIL",
        },
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
