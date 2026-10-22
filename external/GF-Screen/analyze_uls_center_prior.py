#!/usr/bin/env python3
"""Measure whether click-centred ULS+ crops encode a deployment-limiting prior.

This diagnostic reads only the label-free ULS+ inference manifest and predicted
crop masks.  A network that emits foreground near the crop centre for most
inputs is unsafe to run as an unguided dense sliding-window detector: every
window would be encouraged to create a central component.  No ground-truth,
public metric, hidden image, packaging, or upload path exists here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk


FILE_ENDING = ".nii.gz"


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {name: None for name in ("min", "p10", "p25", "p50", "p75", "p90", "max", "mean")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def local_click_zyx(record: dict[str, object]) -> np.ndarray:
    click = np.asarray(record["click_zyx"], dtype=np.int64)
    start_xyz = np.asarray(record["start_xyz"], dtype=np.int64)
    if click.shape != (3,) or start_xyz.shape != (3,):
        raise ValueError("invalid click/start coordinates")
    return click - start_xyz[::-1]


def mask_statistics(mask: np.ndarray, spacing_zyx: np.ndarray, click: np.ndarray) -> dict[str, object]:
    if mask.ndim != 3 or any(size <= 0 for size in mask.shape):
        raise ValueError("ULS+ prediction must be a nonempty 3-D array")
    shape = np.asarray(mask.shape, dtype=np.int64)
    if click.shape != (3,) or np.any(click < 0) or np.any(click >= shape):
        raise ValueError(f"click outside crop: {click.tolist()} versus {shape.tolist()}")
    foreground = np.argwhere(mask)
    voxels = int(foreground.shape[0])
    if voxels == 0:
        return {
            "foreground_voxels": 0,
            "foreground_fraction": 0.0,
            "click_positive": False,
            "central_quarter_positive": False,
            "touches_crop_boundary": False,
            "nearest_foreground_to_click_mm": None,
            "centroid_to_click_mm": None,
            "centroid_normalized_radius": None,
        }

    delta_mm = (foreground - click[None, :]) * spacing_zyx[None, :]
    nearest = float(np.sqrt(np.min(np.sum(delta_mm * delta_mm, axis=1))))
    centroid = foreground.mean(axis=0)
    centroid_mm = float(np.linalg.norm((centroid - click) * spacing_zyx))
    half_extent = np.maximum((shape - 1) / 2.0, 1.0)
    normalized = float(np.linalg.norm((centroid - click) / half_extent))
    quarter = np.maximum(shape // 8, 1)
    lo = np.maximum(click - quarter, 0)
    hi = np.minimum(click + quarter + 1, shape)
    central = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    touches = any(
        np.any(foreground[:, axis] == 0) or np.any(foreground[:, axis] == shape[axis] - 1)
        for axis in range(3)
    )
    return {
        "foreground_voxels": voxels,
        "foreground_fraction": float(voxels / mask.size),
        "click_positive": bool(mask[tuple(int(v) for v in click)]),
        "central_quarter_positive": bool(np.any(mask[central])),
        "touches_crop_boundary": bool(touches),
        "nearest_foreground_to_click_mm": nearest,
        "centroid_to_click_mm": centroid_mm,
        "centroid_normalized_radius": normalized,
    }


def analyze(manifest: Path, predictions: Path, expected_count: int | None) -> dict[str, object]:
    records = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("ULS+ manifest must be a nonempty list")
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, found {len(records)}")
    ids = [str(record["crop_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ULS+ crop IDs")
    expected = {f"{crop_id}{FILE_ENDING}" for crop_id in ids}
    observed = {path.name for path in predictions.glob(f"*{FILE_ENDING}")}
    if expected != observed:
        raise ValueError(
            f"prediction inventory mismatch: missing={len(expected-observed)}, extra={len(observed-expected)}"
        )

    per_crop: list[dict[str, object]] = []
    for record in records:
        crop_id = str(record["crop_id"])
        path = predictions / f"{crop_id}{FILE_ENDING}"
        image = sitk.ReadImage(str(path))
        mask = sitk.GetArrayFromImage(image) > 0
        spacing_zyx = np.asarray(tuple(reversed(image.GetSpacing())), dtype=np.float64)
        click = local_click_zyx(record)
        stats = mask_statistics(mask, spacing_zyx, click)
        stats.update({"crop_id": crop_id, "case": str(record["case"])})
        per_crop.append(stats)

    nonempty = [row for row in per_crop if int(row["foreground_voxels"]) > 0]
    count = len(per_crop)
    nonempty_count = len(nonempty)
    rate = lambda key, rows=per_crop: float(sum(bool(row[key]) for row in rows) / len(rows)) if rows else 0.0
    payload: dict[str, object] = {
        "schema_version": 1,
        "selection_basis": "label-free ULS+ crop-output centre-prior audit",
        "manifest": str(manifest.resolve()),
        "predictions": str(predictions.resolve()),
        "crops": count,
        "cases": len({str(row["case"]) for row in per_crop}),
        "nonempty_crops": nonempty_count,
        "nonempty_rate": float(nonempty_count / count),
        "all_crop_click_positive_rate": rate("click_positive"),
        "nonempty_click_positive_rate": rate("click_positive", nonempty),
        "all_crop_central_quarter_positive_rate": rate("central_quarter_positive"),
        "nonempty_central_quarter_positive_rate": rate("central_quarter_positive", nonempty),
        "nonempty_boundary_touch_rate": rate("touches_crop_boundary", nonempty),
        "foreground_fraction": quantiles(float(row["foreground_fraction"]) for row in nonempty),
        "nearest_foreground_to_click_mm": quantiles(
            float(row["nearest_foreground_to_click_mm"]) for row in nonempty
        ),
        "centroid_to_click_mm": quantiles(float(row["centroid_to_click_mm"]) for row in nonempty),
        "centroid_normalized_radius": quantiles(
            float(row["centroid_normalized_radius"]) for row in nonempty
        ),
        "dense_sliding_safety_interpretation": (
            "UNSAFE_STRONG_CENTER_PRIOR"
            if nonempty_count and rate("central_quarter_positive", nonempty) >= 0.90
            else "CENTER_PRIOR_NOT_ESTABLISHED"
        ),
        "public_labels_read": False,
        "hidden_data_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    for value in payload.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite summary statistic")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    payload = analyze(args.manifest, args.predictions, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
