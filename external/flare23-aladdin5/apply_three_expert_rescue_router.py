#!/usr/bin/env python3
"""Apply the frozen FLARE23 rescue router above a frozen GF/FLARE25 route."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

from train_three_expert_rescue_router import (
    FEATURES,
    extract_one,
    load_confidence,
    sha256,
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def predict(report: dict, x: np.ndarray) -> np.ndarray:
    model = report["final_model"]
    if model["features"] != list(FEATURES):
        raise RuntimeError("rescue feature contract mismatch")
    if model["spec"]["family"] == "ridge":
        mean = np.asarray(model["scaler_mean"], dtype=float)
        scale = np.asarray(model["scaler_scale"], dtype=float)
        coefficients = np.asarray(model["coefficients"], dtype=float)
        return ((x - mean) / scale) @ coefficients + float(model["intercept"])
    model_file = Path(model["pickle"]["path"])
    if sha256(model_file) != model["pickle"]["sha256"]:
        raise RuntimeError("rescue pickle hash mismatch")
    return pickle.loads(model_file.read_bytes()).predict(x)


def read_stage1(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["case"]: row for row in csv.DictReader(stream)}
    for row in rows.values():
        if "choose_flare" not in row and "choose_flare25" in row:
            row["choose_flare"] = row["choose_flare25"]
        if "choose_flare25" not in row and "choose_flare" in row:
            row["choose_flare25"] = row["choose_flare"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--f25", required=True, type=Path)
    parser.add_argument("--f23", required=True, type=Path)
    parser.add_argument("--stage1-routing", required=True, type=Path)
    parser.add_argument("--confidence-terminal", required=True, action="append", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        raise RuntimeError("rescue report is incomplete")
    inventories = [{path.name for path in directory.glob("*.nii.gz")}
                   for directory in (args.gf, args.f25, args.f23)]
    if any(names != inventories[0] for names in inventories[1:]) or len(inventories[0]) != args.expected:
        raise RuntimeError(f"expert inventory mismatch: {[len(names) for names in inventories]}")
    stage1 = read_stage1(args.stage1_routing)
    confidence = load_confidence(args.confidence_terminal)
    case_names = {name.removesuffix(".nii.gz") for name in inventories[0]}
    if set(stage1) != case_names or set(confidence) != case_names:
        raise RuntimeError("routing/confidence inventory mismatch")
    items = [{"name": name, "prediction_name": name + ".nii.gz"}
             for name in sorted(case_names)]
    tasks = [(item, str(args.gf), str(args.f25), str(args.f23), stage1[item["name"]],
              confidence[item["name"]]) for item in items]
    extracted = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, values) in enumerate(pool.map(extract_one, tasks, chunksize=1), 1):
            extracted[name] = values
            if index % 25 == 0 or index == args.expected:
                print(f"rescue features {index}/{args.expected}", flush=True)
    names = [item["name"] for item in items]
    x = np.asarray([extracted[name] for name in names], dtype=float)
    predicted_gain = predict(report, x)
    threshold = float(report["final_model"]["threshold"])
    choose_f23 = predicted_gain > threshold
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rescue in zip(names, choose_f23):
        row = stage1[name]
        if rescue:
            source = args.f23 / f"{name}.nii.gz"
        elif int(row["choose_flare25"]) > 0:
            source = args.f25 / f"{name}.nii.gz"
        else:
            source = args.gf / f"{name}.nii.gz"
        shutil.copyfile(source, args.output / f"{name}.nii.gz")
    routing_csv = args.output.parent / f"{args.output.name}.routing.csv"
    with routing_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = ["case", "stage1_choose_f25", "choose_f23", "predicted_rescue_gain",
                  "threshold", *FEATURES]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow({"case": name,
                             "stage1_choose_f25": int(stage1[name]["choose_flare25"]),
                             "choose_f23": int(choose_f23[i]),
                             "predicted_rescue_gain": float(predicted_gain[i]),
                             "threshold": threshold, **dict(zip(FEATURES, x[i]))})
    payload = {
        "schema_version": 1, "status": "COMPLETE", "cases": len(names),
        "stage1_switches_to_f25": int(sum(int(stage1[name]["choose_flare25"]) for name in names)),
        "stage2_switches_to_f23": int(choose_f23.sum()), "threshold": threshold,
        "report": {"path": str(args.report.resolve()), "sha256": sha256(args.report)},
        "routing_csv": {"path": str(routing_csv.resolve()), "sha256": sha256(routing_csv)},
        "gf": str(args.gf.resolve()), "f25": str(args.f25.resolve()),
        "f23": str(args.f23.resolve()), "output": str(args.output.resolve()),
        "labels_read": False, "uploaded": False,
    }
    atomic_json(args.output.parent / f"{args.output.name}.routing.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
