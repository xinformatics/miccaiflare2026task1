#!/usr/bin/env python3
"""Apply a frozen exact-707 mask router without labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_exact707_mask_router import FEATURES, extract_one  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(report: dict, x: np.ndarray) -> np.ndarray:
    model = report["final_model"]
    if model["features"] != list(FEATURES):
        raise RuntimeError("router feature contract mismatch")
    if model["spec"]["family"] == "ridge":
        mean = np.asarray(model["scaler_mean"], dtype=float)
        scale = np.asarray(model["scaler_scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        return ((x - mean) / scale) @ coefficients + float(model["intercept"])
    model_file = Path(model["pickle"]["path"])
    if sha256(model_file) != model["pickle"]["sha256"]:
        raise RuntimeError("router pickle hash mismatch")
    return pickle.loads(model_file.read_bytes()).predict(x)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--flare", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("status") != "COMPLETE" or not report.get("router_audit_gate_pass"):
        raise RuntimeError("router audit did not pass")
    gf_names = {path.name for path in args.gf.glob("*.nii.gz")}
    flare_names = {path.name for path in args.flare.glob("*.nii.gz")}
    if gf_names != flare_names or len(gf_names) != args.expected:
        raise RuntimeError(f"mask inventory mismatch: GF={len(gf_names)} FLARE={len(flare_names)}")
    names = sorted(gf_names)
    tasks = [({"name": name.removesuffix(".nii.gz"), "prediction_name": name},
              str(args.gf), str(args.flare)) for name in names]
    features: dict[str, list[float]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, values in pool.map(extract_one, tasks, chunksize=1):
            features[name + ".nii.gz"] = values
    x = np.asarray([features[name] for name in names], dtype=float)
    predicted_gain = predict(report, x)
    threshold = float(report["final_model"]["threshold"])
    choose_flare = predicted_gain > threshold
    args.output.mkdir(parents=True, exist_ok=True)
    for name, choose in zip(names, choose_flare):
        source = (args.flare if choose else args.gf) / name
        shutil.copyfile(source, args.output / name)
    decisions = args.output.parent / f"{args.output.name}.routing.csv"
    with decisions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "choose_flare25", "predicted_gain", "threshold", *FEATURES])
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow({"case": name.removesuffix(".nii.gz"),
                             "choose_flare25": int(choose_flare[index]),
                             "predicted_gain": float(predicted_gain[index]),
                             "threshold": threshold,
                             **dict(zip(FEATURES, x[index]))})
    summary = {
        "schema_version": 1, "status": "COMPLETE", "cases": len(names),
        "switches_to_flare25": int(choose_flare.sum()), "threshold": threshold,
        "router_report": {"path": str(args.report.resolve()), "sha256": sha256(args.report)},
        "gf": str(args.gf.resolve()), "flare25": str(args.flare.resolve()),
        "output": str(args.output.resolve()),
        "routing_csv": {"path": str(decisions.resolve()), "sha256": sha256(decisions)},
        "labels_read": False, "uploaded": False,
    }
    atomic_json(args.output.parent / f"{args.output.name}.routing.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
