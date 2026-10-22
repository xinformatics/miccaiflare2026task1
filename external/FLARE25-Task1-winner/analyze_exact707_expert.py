#!/usr/bin/env python3
"""Positive-only comparison of FLARE25 and GF on the frozen exact-707 set."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_mask(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"nonfinite mask: {path}")
    if reference is not None:
        if image.shape != reference.shape:
            raise ValueError(f"shape mismatch: {path}")
        if not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-4):
            raise ValueError(f"affine mismatch: {path}")
        if not np.allclose(image.header.get_zooms()[:3], reference.header.get_zooms()[:3], rtol=0, atol=1e-5):
            raise ValueError(f"spacing mismatch: {path}")
    return image, data > 0


def dice(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = int(target.sum()) + int(prediction.sum())
    return float(2 * np.count_nonzero(target & prediction) / denominator) if denominator else 1.0


def matched_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float, int, int]:
    target_voxels = int(target.sum())
    overlap_voxels = int(np.count_nonzero(target & prediction))
    recall = float(overlap_voxels / target_voxels) if target_voxels else 1.0
    points = np.argwhere(prediction)
    if points.size == 0:
        return dice(target, prediction), recall, 0, 0
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    prediction_crop = prediction[crop]
    target_crop = target[crop]
    component_map, count = ndimage.label(prediction_crop, structure=CONNECTIVITY_26)
    touching = np.unique(component_map[target_crop])
    touching = touching[touching > 0]
    if touching.size:
        matched = np.isin(component_map, touching)
        matched_voxels = int(matched.sum())
        intersection = int(np.count_nonzero(target_crop & matched))
        denominator = target_voxels + matched_voxels
        matched_dice = float(2 * intersection / denominator) if denominator else 1.0
    else:
        matched_dice = 0.0 if target_voxels else 1.0
    return matched_dice, recall, int(touching.size), int(count)


def evaluate_item(task: tuple[dict, str, str, dict | None]) -> dict:
    """Evaluate one independent case; safe to execute in a worker process."""
    item, flare_directory, gf_directory, reused_gf = task
    name = item["prediction_name"]
    label_image, target = load_mask(Path(item["label"]))
    _, flare = load_mask(Path(flare_directory) / name, label_image)
    flare_dice, flare_recall, flare_touching, flare_components = matched_metrics(target, flare)
    if reused_gf is None:
        _, gf = load_mask(Path(gf_directory) / name, label_image)
        gf_dice, gf_recall, gf_touching, gf_components = matched_metrics(target, gf)
    else:
        gf_dice = float(reused_gf["gf_matched_dice"])
        gf_recall = float(reused_gf["gf_target_recall"])
        gf_touching = int(reused_gf["gf_touching_components"])
        gf_components = int(reused_gf["gf_total_components"])
    return {
        "case": item["name"],
        "cohort": item["cohort"],
        "flare25_matched_dice": flare_dice,
        "gf_matched_dice": gf_dice,
        "oracle_matched_dice": max(flare_dice, gf_dice),
        "flare25_target_recall": flare_recall,
        "gf_target_recall": gf_recall,
        "flare25_wins": flare_dice > gf_dice,
        "flare25_touching_components": flare_touching,
        "gf_touching_components": gf_touching,
        "flare25_total_components": flare_components,
        "gf_total_components": gf_components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--flare", required=True, type=Path)
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reuse-gf-cases",
        type=Path,
        help="Previously validated exact-707 cases.csv from the same GF masks",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    items = manifest["items"]
    if len(items) != 707 or manifest.get("cases") != 707:
        raise RuntimeError("frozen manifest is not exact-707")
    flare_files = list(args.flare.glob("*.nii.gz"))
    gf_files = list(args.gf.glob("*.nii.gz"))
    if len(flare_files) != 707 or len(gf_files) != 707:
        raise RuntimeError(f"prediction inventory must be 707/707, got {len(flare_files)}/{len(gf_files)}")

    reused = None
    if args.reuse_gf_cases is not None:
        with args.reuse_gf_cases.open(newline="", encoding="utf-8") as stream:
            reused = {row["case"]: row for row in csv.DictReader(stream)}
        if len(reused) != 707 or {item["name"] for item in items} != set(reused):
            raise RuntimeError("reused GF metric inventory is not exact-707")
    tasks = (
        (item, str(args.flare), str(args.gf), None if reused is None else reused[item["name"]])
        for item in items
    )
    rows: list[dict] = []
    if args.workers == 1:
        iterator = map(evaluate_item, tasks)
        for index, row in enumerate(iterator, start=1):
            rows.append(row)
            if index % 25 == 0 or index == len(items):
                print(f"{index}/{len(items)} primary-positive cases", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            iterator = pool.map(evaluate_item, tasks, chunksize=1)
            for index, row in enumerate(iterator, start=1):
                rows.append(row)
                if index % 25 == 0 or index == len(items):
                    print(
                        f"{index}/{len(items)} primary-positive cases "
                        f"with {args.workers} workers",
                        flush=True,
                    )

    by_cohort: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["cohort"]].append(row)
    for cohort, group in sorted(grouped.items()):
        flare = np.asarray([row["flare25_matched_dice"] for row in group])
        gf = np.asarray([row["gf_matched_dice"] for row in group])
        by_cohort[cohort] = {
            "cases": len(group),
            "flare25_mean_matched_dice": float(flare.mean()),
            "gf_mean_matched_dice": float(gf.mean()),
            "flare25_case_win_fraction": float(np.mean(flare > gf)),
            "oracle_gain_over_gf": float(np.maximum(flare, gf).mean() - gf.mean()),
        }
    flare = np.asarray([row["flare25_matched_dice"] for row in rows])
    gf = np.asarray([row["gf_matched_dice"] for row in rows])
    win_fraction = float(np.mean(flare > gf))
    oracle_gain = float(np.maximum(flare, gf).mean() - gf.mean())
    strong_cohorts = sum(value["oracle_gain_over_gf"] >= 0.02 for value in by_cohort.values())
    gate = policy["router_feasibility_gate"]
    checks = {
        "flare25_case_win_fraction": win_fraction >= float(gate["minimum_flare25_case_win_fraction"]),
        "oracle_mean_dice_gain": oracle_gain >= float(gate["minimum_oracle_mean_dice_gain_over_gf"]),
        "cohort_replication": strong_cohorts >= int(gate["minimum_cohorts_with_oracle_gain_at_least_0p02"]),
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_role": "PRIMARY_LESION_EXPERT_DIAGNOSTIC_ONLY",
        "cases": len(rows),
        "flare25_mean_matched_dice": float(flare.mean()),
        "gf_mean_matched_dice": float(gf.mean()),
        "flare25_case_win_fraction": win_fraction,
        "oracle_mean_matched_dice": float(np.maximum(flare, gf).mean()),
        "oracle_gain_over_gf": oracle_gain,
        "cohorts_with_oracle_gain_at_least_0p02": strong_cohorts,
        "by_cohort": by_cohort,
        "gate_checks": checks,
        "router_feasibility_gate_pass": all(checks.values()),
        "assets": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "policy": {"path": str(args.policy.resolve()), "sha256": sha256(args.policy)},
            "flare25_predictions": str(args.flare.resolve()),
            "gf_predictions": str(args.gf.resolve()),
        },
        "constraints": {
            "partial_labels_positive_only": True,
            "disconnected_predictions_scored_as_false_positive": False,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
            "router_trained": False,
        },
        "execution": {
            "workers": args.workers,
            "gf_metrics_reused": args.reuse_gf_cases is not None,
            "reused_gf_cases": (
                None
                if args.reuse_gf_cases is None
                else {
                    "path": str(args.reuse_gf_cases.resolve()),
                    "sha256": sha256(args.reuse_gf_cases),
                }
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(args.output / "summary.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
