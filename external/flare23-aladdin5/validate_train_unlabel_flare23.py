#!/usr/bin/env python3
"""Validate a sharded FLARE23 Train-Unlabel proposal inventory."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--terminal", required=True, action="append", type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    source_names = sorted(path.name for path in args.images.glob("*_0000.nii.gz"))
    expected_names = {name.replace("_0000.nii.gz", ".nii.gz") for name in source_names}
    prediction_paths = sorted(args.predictions.glob("*.nii.gz"))
    prediction_names = {path.name for path in prediction_paths}
    if len(source_names) != args.expected or prediction_names != expected_names:
        raise RuntimeError(
            f"inventory mismatch: sources={len(source_names)} predictions={len(prediction_names)}"
        )

    records = []
    for path in args.terminal:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("dataset_role") != "training":
            raise RuntimeError(f"invalid shard terminal: {path}")
        if payload.get("public_labels_read") or payload.get("hidden_read"):
            raise RuntimeError(f"invalid data provenance in {path}")
        records.extend(payload.get("cases", []))
    record_names = [record["case"] for record in records]
    if len(records) != args.expected or len(set(record_names)) != args.expected:
        raise RuntimeError("terminal case inventory is incomplete or duplicated")
    if set(record_names) != prediction_names:
        raise RuntimeError("terminal and prediction inventories differ")

    empty = []
    foreground = []
    geometry_errors = []
    range_errors = []
    for index, prediction_path in enumerate(prediction_paths, start=1):
        image_path = args.images / prediction_path.name.replace(".nii.gz", "_0000.nii.gz")
        prediction_image = nib.load(str(prediction_path))
        source_image = nib.load(str(image_path))
        if prediction_image.shape != source_image.shape or not np.allclose(
            prediction_image.affine, source_image.affine, rtol=0, atol=1e-4
        ):
            geometry_errors.append(prediction_path.name)
        prediction = np.asanyarray(prediction_image.dataobj)
        lower = int(prediction.min())
        upper = int(prediction.max())
        if lower < 0 or upper > 1:
            range_errors.append([prediction_path.name, lower, upper])
        count = int(np.count_nonzero(prediction))
        foreground.append(count)
        if count == 0:
            empty.append(prediction_path.name)
        del prediction
        if index % 250 == 0 or index == args.expected:
            print(f"validation progress {index}/{args.expected}", flush=True)
    if geometry_errors or range_errors:
        raise RuntimeError(
            f"content validation failed: geometry={len(geometry_errors)} range={len(range_errors)}"
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "cases": args.expected,
        "unique_cases": len(set(record_names)),
        "geometry_match": True,
        "binary_masks": True,
        "empty_masks": len(empty),
        "empty_case_names": empty,
        "foreground_voxels_min": min(foreground),
        "foreground_voxels_median": float(np.median(foreground)),
        "foreground_voxels_max": max(foreground),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
