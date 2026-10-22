#!/usr/bin/env python3
"""Public-only lesion appearance/geometry audit for incumbent failure cases.

The audit is diagnostic. It reads the fixed public images, labels, and an
already-materialized per-case incumbent score table. It does not train,
select a deployment, inspect hidden data, or create a submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_case(name: str) -> str:
    name = Path(name).name
    if name.endswith("_0000.nii.gz"):
        return name[:-12] + ".nii.gz"
    return name if name.endswith(".nii.gz") else name + ".nii.gz"


def finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def quantile(values: np.ndarray, probability: float) -> float:
    values = finite(values)
    return float(np.quantile(values, probability)) if values.size else float("nan")


def body_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied = [np.flatnonzero(np.any(mask, axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(axis.size == 0 for axis in occupied):
        return np.zeros(3, dtype=int), np.asarray(mask.shape, dtype=int) - 1
    return (
        np.asarray([axis[0] for axis in occupied], dtype=int),
        np.asarray([axis[-1] for axis in occupied], dtype=int),
    )


def read_scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"case", "dice"}.issubset(rows[0]):
        raise ValueError("score table must contain case,dice")
    result: dict[str, float] = {}
    for row in rows:
        name = canonical_case(row["case"])
        if name in result:
            raise ValueError(f"duplicate score case: {name}")
        result[name] = float(row["dice"])
    return result


def case_features(image_path: Path, label_path: Path, dice: float) -> dict[str, float | int | str | bool]:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(f"geometry size mismatch: {image_path.name}")
    if not np.allclose(image.GetSpacing(), label.GetSpacing(), rtol=0, atol=1e-6):
        raise ValueError(f"geometry spacing mismatch: {image_path.name}")
    ct = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    lesion = sitk.GetArrayFromImage(label) > 0
    if not lesion.any():
        raise ValueError(f"public case has empty lesion label: {image_path.name}")
    spacing = np.asarray(image.GetSpacing()[::-1], dtype=np.float64)
    voxel_mm3 = float(np.prod(spacing))
    components, count = ndimage.label(lesion, structure=np.ones((3, 3, 3), dtype=np.uint8))
    component_voxels = np.bincount(components.ravel())[1:]
    component_mm3 = component_voxels.astype(np.float64) * voxel_mm3

    coordinates = np.argwhere(lesion)
    lesion_lower = coordinates.min(axis=0)
    lesion_upper = coordinates.max(axis=0)
    lesion_centroid = coordinates.mean(axis=0)
    body = np.isfinite(ct) & (ct > -500) & (ct < 3000)
    lower, upper = body_bounds(body)
    denominator = np.maximum(upper - lower, 1)
    normalized_centroid = (lesion_centroid - lower) / denominator

    pad = np.ceil(20.0 / spacing).astype(int)
    start = np.maximum(lesion_lower - pad, 0)
    stop = np.minimum(lesion_upper + pad + 1, np.asarray(ct.shape))
    region = tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))
    local_lesion = lesion[region]
    distance = ndimage.distance_transform_edt(~local_lesion, sampling=spacing)
    shell = (distance >= 3.0) & (distance <= 15.0)
    shell_values = finite(ct[region][shell])
    shell_values = shell_values[(shell_values > -500) & (shell_values < 1000)]
    lesion_values = finite(ct[lesion])
    lesion_median = quantile(lesion_values, 0.5)
    shell_median = quantile(shell_values, 0.5)
    shell_std = float(np.std(shell_values)) if shell_values.size else float("nan")
    signed_contrast = lesion_median - shell_median
    cnr = abs(signed_contrast) / max(shell_std, 1e-6)

    soft_tissue = finite(ct[body])
    return {
        "case": canonical_case(image_path.name),
        "incumbent_dice": float(dice),
        "zero_dice": bool(dice == 0.0),
        "shape_z": int(ct.shape[0]),
        "shape_y": int(ct.shape[1]),
        "shape_x": int(ct.shape[2]),
        "spacing_z_mm": float(spacing[0]),
        "spacing_y_mm": float(spacing[1]),
        "spacing_x_mm": float(spacing[2]),
        "lesion_components": int(count),
        "lesion_total_mm3": float(lesion.sum() * voxel_mm3),
        "lesion_largest_mm3": float(component_mm3.max()),
        "lesion_median_component_mm3": float(np.median(component_mm3)),
        "lesion_extent_z_mm": float((lesion_upper[0] - lesion_lower[0] + 1) * spacing[0]),
        "lesion_extent_y_mm": float((lesion_upper[1] - lesion_lower[1] + 1) * spacing[1]),
        "lesion_extent_x_mm": float((lesion_upper[2] - lesion_lower[2] + 1) * spacing[2]),
        "body_relative_z": float(normalized_centroid[0]),
        "body_relative_y": float(normalized_centroid[1]),
        "body_relative_x": float(normalized_centroid[2]),
        "lesion_hu_mean": float(np.mean(lesion_values)),
        "lesion_hu_median": lesion_median,
        "lesion_hu_std": float(np.std(lesion_values)),
        "lesion_hu_iqr": quantile(lesion_values, 0.75) - quantile(lesion_values, 0.25),
        "local_shell_hu_median": shell_median,
        "local_shell_hu_std": shell_std,
        "signed_local_contrast_hu": signed_contrast,
        "absolute_local_contrast_hu": abs(signed_contrast),
        "local_contrast_to_noise": cnr,
        "body_hu_median": quantile(soft_tissue, 0.5),
        "body_hu_iqr": quantile(soft_tissue, 0.75) - quantile(soft_tissue, 0.25),
    }


def comparison(rows: list[dict[str, object]], feature: str) -> dict[str, object]:
    zero = np.asarray([float(row[feature]) for row in rows if row["zero_dice"]], dtype=float)
    other = np.asarray([float(row[feature]) for row in rows if not row["zero_dice"]], dtype=float)
    zero = zero[np.isfinite(zero)]
    other = other[np.isfinite(other)]
    all_values = np.asarray([float(row[feature]) for row in rows], dtype=float)
    all_dice = np.asarray([float(row["incumbent_dice"]) for row in rows], dtype=float)
    valid = np.isfinite(all_values)
    u = stats.mannwhitneyu(zero, other, alternative="two-sided") if zero.size and other.size else None
    rho = stats.spearmanr(all_values[valid], all_dice[valid]) if valid.sum() >= 3 else None
    return {
        "zero_median": float(np.median(zero)) if zero.size else None,
        "nonzero_median": float(np.median(other)) if other.size else None,
        "median_difference_zero_minus_nonzero": float(np.median(zero) - np.median(other)) if zero.size and other.size else None,
        "mann_whitney_p": float(u.pvalue) if u is not None else None,
        "spearman_rho_with_dice": float(rho.statistic) if rho is not None else None,
        "spearman_p": float(rho.pvalue) if rho is not None else None,
    }


def main() -> None:
    args = parse_args()
    scores = read_scores(args.scores)
    image_paths = sorted(args.images.glob("*.nii.gz"))
    label_paths = sorted(args.labels.glob("*.nii.gz"))
    if len(image_paths) != 50 or len(label_paths) != 50:
        raise SystemExit("fixed public inventory must contain exactly 50 images and labels")
    image_by_case = {canonical_case(path.name): path for path in image_paths}
    label_by_case = {canonical_case(path.name): path for path in label_paths}
    if len(image_by_case) != 50 or len(label_by_case) != 50:
        raise SystemExit("canonical public case identifiers are not unique")
    if set(image_by_case) != set(scores) or set(label_by_case) != set(scores):
        raise SystemExit("public image/label/score case inventories differ")

    rows = []
    for index, case in enumerate(sorted(scores), start=1):
        image_path = image_by_case[case]
        print(f"covariate audit {index}/50: {image_path.name}", flush=True)
        rows.append(case_features(image_path, label_by_case[case], scores[case]))

    features = [
        "lesion_components",
        "lesion_total_mm3",
        "lesion_largest_mm3",
        "lesion_median_component_mm3",
        "lesion_extent_z_mm",
        "lesion_extent_y_mm",
        "lesion_extent_x_mm",
        "body_relative_z",
        "body_relative_y",
        "body_relative_x",
        "lesion_hu_median",
        "lesion_hu_std",
        "lesion_hu_iqr",
        "signed_local_contrast_hu",
        "absolute_local_contrast_hu",
        "local_contrast_to_noise",
        "body_hu_median",
        "body_hu_iqr",
        "spacing_z_mm",
    ]
    comparisons = {feature: comparison(rows, feature) for feature in features}
    args.output.mkdir(parents=True, exist_ok=True)
    cases_path = args.output / "cases.csv"
    with cases_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "purpose": "public-only incumbent zero-Dice lesion appearance and geometry audit",
        "case_count": len(rows),
        "zero_dice_cases": [row["case"] for row in rows if row["zero_dice"]],
        "zero_dice_count": sum(bool(row["zero_dice"]) for row in rows),
        "comparisons": comparisons,
        "cases": {"path": str(cases_path.resolve()), "sha256": sha256(cases_path)},
        "interpretation_guard": "small-sample public diagnostic only; not a deployment selector or threshold sweep",
        "public_images_read": True,
        "public_labels_read": True,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
