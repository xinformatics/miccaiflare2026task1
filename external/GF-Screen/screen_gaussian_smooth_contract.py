#!/usr/bin/env python3
"""Screen physical-space Gaussian boundary smoothing and contraction.

This is a validation-only post-processing diagnostic. Each connected component
is filtered inside a padded crop, which avoids allocating a float volume for an
entire CT and prevents disconnected lesions from influencing one another.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
from evaluate import score_case  # noqa: E402


# sigma is specified in physical millimetres. Threshold >0.5 contracts the
# smoothed boundary; identity is retained as an exact implementation control.
CONFIGS = (
    ("identity", 0.0, 0.50),
    ("smooth_s050_t050", 0.5, 0.50),
    ("smooth_s075_t050", 0.75, 0.50),
    ("smooth_s100_t050", 1.0, 0.50),
    ("smooth_s100_t055", 1.0, 0.55),
    ("smooth_s100_t060", 1.0, 0.60),
    ("smooth_s100_t065", 1.0, 0.65),
    ("smooth_s100_t070", 1.0, 0.70),
    ("smooth_s150_t050", 1.5, 0.50),
    ("smooth_s150_t055", 1.5, 0.55),
    ("smooth_s150_t060", 1.5, 0.60),
    ("smooth_s150_t065", 1.5, 0.65),
    ("smooth_s150_t070", 1.5, 0.70),
    ("smooth_s200_t050", 2.0, 0.50),
    ("smooth_s200_t055", 2.0, 0.55),
    ("smooth_s200_t060", 2.0, 0.60),
    ("smooth_s200_t065", 2.0, 0.65),
    ("smooth_s200_t070", 2.0, 0.70),
)


def smooth_components(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    sigma_mm: float,
    threshold: float,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if sigma_mm == 0:
        return mask.copy()
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
    output = np.zeros_like(mask)
    sigma_vox = np.asarray([sigma_mm / value for value in spacing], dtype=np.float64)
    pad_width = np.maximum(1, np.ceil(4.0 * sigma_vox).astype(int))
    padding = tuple((int(value), int(value)) for value in pad_width)
    for component_id, component_slice in enumerate(ndimage.find_objects(labels), 1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        working = np.pad(component.astype(np.float32), padding, mode="constant")
        softened = ndimage.gaussian_filter(
            working, sigma=sigma_vox, mode="constant", cval=0.0, truncate=4.0
        )
        transformed = softened >= threshold
        transformed = transformed[
            tuple(
                slice(before, transformed.shape[axis] - after)
                for axis, (before, after) in enumerate(padding)
            )
        ]
        output[component_slice] |= transformed
    return output


def softened_component_crops(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    sigma_mm: float,
) -> list[tuple[tuple[slice, ...], np.ndarray]]:
    """Filter every component once so several thresholds can share the result."""
    labels = cc3d.connected_components(np.asarray(mask, dtype=np.uint8), connectivity=26)
    sigma_vox = np.asarray([sigma_mm / value for value in spacing], dtype=np.float64)
    pad_width = np.maximum(1, np.ceil(4.0 * sigma_vox).astype(int))
    padding = tuple((int(value), int(value)) for value in pad_width)
    softened_components = []
    for component_id, component_slice in enumerate(ndimage.find_objects(labels), 1):
        if component_slice is None:
            continue
        component = labels[component_slice] == component_id
        working = np.pad(component.astype(np.float32), padding, mode="constant")
        softened = ndimage.gaussian_filter(
            working, sigma=sigma_vox, mode="constant", cval=0.0, truncate=4.0
        )
        softened = softened[
            tuple(
                slice(before, softened.shape[axis] - after)
                for axis, (before, after) in enumerate(padding)
            )
        ]
        softened_components.append((component_slice, softened))
    return softened_components


def score_one(
    item: tuple[str, str, str, tuple[str, ...], bool]
) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, prediction_path, target_path, selected_names, dsc_only = item
    prediction_image = nib.load(prediction_path)
    target_image = nib.load(target_path)
    if prediction_image.shape != target_image.shape or not np.allclose(
        prediction_image.affine, target_image.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")
    prediction = np.asanyarray(prediction_image.dataobj).astype(bool)
    target = np.asanyarray(target_image.dataobj).astype(bool)
    spacing = tuple(float(value) for value in target_image.header.get_zooms()[:3])
    values = []

    def append_score(config: str, transformed: np.ndarray) -> None:
        if dsc_only:
            denominator = int(target.sum()) + int(transformed.sum())
            dsc = (
                1.0
                if denominator == 0
                else 2.0 * float(np.logical_and(target, transformed).sum()) / denominator
            )
            nsd = float("nan")
        else:
            dsc, nsd = score_case(
                target.astype(np.uint8), transformed.astype(np.uint8), spacing
            )
        values.append((config, float(dsc), float(nsd), int(transformed.sum())))

    selected_configs = [item for item in CONFIGS if item[0] in selected_names]
    if any(config == "identity" for config, _, _ in selected_configs):
        append_score("identity", prediction)
    for sigma_mm in sorted({item[1] for item in selected_configs if item[1] > 0}):
        softened = softened_component_crops(prediction, spacing, sigma_mm)
        for config, _, threshold in [item for item in selected_configs if item[1] == sigma_mm]:
            transformed = np.zeros_like(prediction)
            for component_slice, component_values in softened:
                transformed[component_slice] |= component_values >= threshold
            append_score(config, transformed)
    return name.removesuffix(".nii.gz"), values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--dsc-only", action="store_true", help="fast first-stage screen without NSD"
    )
    parser.add_argument(
        "--only-config", action="append", choices=[item[0] for item in CONFIGS]
    )
    args = parser.parse_args()
    names = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if len(names) != 50:
        parser.error(f"expected exact fixed public inventory of 50, found {len(names)}")
    if args.workers < 1 or args.workers > 2:
        parser.error("workers must be 1 or 2 for the memory-safe screen")
    missing = [name for name in names if not (args.predictions / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing predictions; first={missing[0]}")

    args.output.mkdir(parents=True, exist_ok=False)
    selected_names = tuple(args.only_config or [item[0] for item in CONFIGS])
    selected_configs = [item for item in CONFIGS if item[0] in selected_names]
    rows = {config[0]: [] for config in selected_configs}
    items = [
        (
            name,
            str(args.predictions / name),
            str(args.labels / name),
            selected_names,
            args.dsc_only,
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, values) in enumerate(executor.map(score_one, items), 1):
            for config, dsc, nsd, foreground in values:
                rows[config].append((case, dsc, nsd, foreground))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)

    summary = []
    for config, sigma_mm, threshold in selected_configs:
        path = args.output / f"{config}.metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
            writer.writerows(rows[config])
        dsc = float(np.mean([row[1] for row in rows[config]]))
        nsd = float(np.mean([row[2] for row in rows[config]]))
        foreground = int(sum(row[3] for row in rows[config]))
        summary.append(
            {
                "config": config,
                "sigma_mm": sigma_mm,
                "threshold": threshold,
                "DSC": dsc,
                "NSD": nsd,
                "combined": (dsc + nsd) / 2.0,
                "foreground_voxels": foreground,
                "metrics": str(path),
            }
        )
    summary.sort(
        key=lambda row: (
            -row["DSC"],
            -(row["NSD"] if np.isfinite(row["NSD"]) else -1.0),
            row["config"],
        )
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(
            f"{row['config']}\tDSC={row['DSC']:.6f}\tNSD={row['NSD']:.6f}"
            f"\tfg={row['foreground_voxels']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
