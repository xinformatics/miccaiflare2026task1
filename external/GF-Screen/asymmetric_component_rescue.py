#!/usr/bin/env python3
"""Recover prompt-rejected GF components using independent model support.

The primary support keeps the deployed exact-overlap confirmation rule. A base
component rejected by that rule may be restored only when the independent
secondary mask overlaps the *same* component by a predeclared amount. The
operation can never introduce foreground outside the immutable base mask.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "handoff_bundle" / "code"))
from evaluate import score_case  # noqa: E402


CONFIGS: tuple[tuple[str, dict[str, float]], ...] = (
    ("secondary_exact", {}),
    ("secondary_mm10", {"min_overlap_mm3": 10.0}),
    ("secondary_mm30", {"min_overlap_mm3": 30.0}),
    ("secondary_frac001", {"min_overlap_fraction": 0.001}),
    ("secondary_frac005", {"min_overlap_fraction": 0.005}),
    (
        "secondary_mm10_frac001",
        {"min_overlap_mm3": 10.0, "min_overlap_fraction": 0.001},
    ),
    (
        "secondary_mm30_frac005",
        {"min_overlap_mm3": 30.0, "min_overlap_fraction": 0.005},
    ),
)


def load_bool(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def validate_geometry(
    name: str,
    reference: nib.Nifti1Image,
    candidate: nib.Nifti1Image,
) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")


def secondary_keep(
    overlap: np.ndarray,
    component_counts: np.ndarray,
    voxel_mm3: float,
    config: dict[str, float],
) -> np.ndarray:
    keep = overlap > 0
    if "min_overlap_mm3" in config:
        keep &= overlap * voxel_mm3 >= float(config["min_overlap_mm3"])
    if "min_overlap_fraction" in config:
        fraction = np.divide(
            overlap,
            component_counts,
            out=np.zeros_like(overlap, dtype=np.float64),
            where=component_counts > 0,
        )
        keep &= fraction >= float(config["min_overlap_fraction"])
    keep[0] = False
    return keep


def apply_asymmetric_rescue(
    base: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    spacing: tuple[float, float, float],
    config: dict[str, float],
) -> np.ndarray:
    base = np.asarray(base, dtype=bool)
    primary = np.asarray(primary, dtype=bool)
    secondary = np.asarray(secondary, dtype=bool)
    if base.shape != primary.shape or base.shape != secondary.shape:
        raise ValueError("base and support masks must have identical shapes")
    component_map = cc3d.connected_components(base.astype(np.uint8), connectivity=26)
    component_counts = np.bincount(component_map.ravel())
    primary_overlap = np.bincount(
        component_map[base & primary], minlength=len(component_counts)
    )
    secondary_overlap = np.bincount(
        component_map[base & secondary], minlength=len(component_counts)
    )
    keep = primary_overlap > 0
    keep |= secondary_keep(
        secondary_overlap, component_counts, float(np.prod(spacing)), config
    )
    keep[0] = False
    return keep[component_map]


def names_in(directory: Path) -> list[str]:
    names = sorted(path.name for path in directory.glob("*.nii.gz"))
    if not names:
        raise FileNotFoundError(f"no NIfTI masks in {directory}")
    return names


def validate_inputs(names: list[str], directories: tuple[Path, ...]) -> None:
    expected = set(names)
    for directory in directories:
        observed = set(names_in(directory))
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"case mismatch in {directory}: missing={missing[:3]} extra={extra[:3]}"
            )


def process_case(
    item: tuple[str, str, str, str, str | None],
) -> tuple[str, list[tuple[str, float, float, int]]]:
    name, base_path, primary_path, secondary_path, gt_path = item
    base, base_image = load_bool(Path(base_path))
    primary, primary_image = load_bool(Path(primary_path))
    secondary, secondary_image = load_bool(Path(secondary_path))
    validate_geometry(f"primary/{name}", base_image, primary_image)
    validate_geometry(f"secondary/{name}", base_image, secondary_image)
    spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
    target = None
    if gt_path is not None:
        target, target_image = load_bool(Path(gt_path))
        validate_geometry(f"target/{name}", base_image, target_image)
    values: list[tuple[str, float, float, int]] = []
    for tag, config in CONFIGS:
        prediction = apply_asymmetric_rescue(
            base, primary, secondary, spacing, config
        )
        if target is None:
            dsc = nsd = float("nan")
        else:
            dsc, nsd = score_case(
                target.astype(np.uint8), prediction.astype(np.uint8), spacing
            )
        values.append((tag, float(dsc), float(nsd), int(prediction.sum())))
    return name.removesuffix(".nii.gz"), values


def screen(args: argparse.Namespace) -> None:
    names = names_in(args.gt)
    validate_inputs(names, (args.base, args.primary, args.secondary))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "configs").mkdir(exist_ok=True)
    for tag, config in CONFIGS:
        (args.output / "configs" / f"{tag}.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    rows: dict[str, list[tuple[str, float, float, int]]] = {
        tag: [] for tag, _ in CONFIGS
    }
    items = [
        (
            name,
            str(args.base / name),
            str(args.primary / name),
            str(args.secondary / name),
            str(args.gt / name),
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, values) in enumerate(executor.map(process_case, items), 1):
            for tag, dsc, nsd, foreground in values:
                rows[tag].append((case, dsc, nsd, foreground))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)
    summary_rows: list[tuple[str, float, float, float, str, str]] = []
    for tag, config in CONFIGS:
        metrics = args.output / f"{tag}.metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "pred_fg_voxels"))
            for case, dsc, nsd, foreground in rows[tag]:
                writer.writerow((case, round(dsc, 4), round(nsd, 4), foreground))
        mean_dsc = float(np.mean([row[1] for row in rows[tag]]))
        mean_nsd = float(np.mean([row[2] for row in rows[tag]]))
        summary_rows.append(
            (
                tag,
                mean_dsc,
                mean_nsd,
                (mean_dsc + mean_nsd) / 2,
                str(metrics),
                str(args.output / "configs" / f"{tag}.json"),
            )
        )
    summary_rows.sort(key=lambda row: (-row[3], -row[1], -row[2], row[0]))
    with (args.output / "summary.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("tag", "DSC", "NSD", "combined", "metrics", "config"))
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(f"{row[0]}\tDSC={row[1]:.6f}\tNSD={row[2]:.6f}")


def materialize_one(
    item: tuple[str, str, str, str, str, dict[str, float]],
) -> str:
    name, base_path, primary_path, secondary_path, output_path, config = item
    base, base_image = load_bool(Path(base_path))
    primary, primary_image = load_bool(Path(primary_path))
    secondary, secondary_image = load_bool(Path(secondary_path))
    validate_geometry(f"primary/{name}", base_image, primary_image)
    validate_geometry(f"secondary/{name}", base_image, secondary_image)
    spacing = tuple(float(value) for value in base_image.header.get_zooms()[:3])
    prediction = apply_asymmetric_rescue(base, primary, secondary, spacing, config)
    header = base_image.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(prediction.astype(np.uint8), base_image.affine, header),
        output_path,
    )
    return name


def materialize(args: argparse.Namespace) -> None:
    names = names_in(args.base)
    validate_inputs(names, (args.primary, args.secondary))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    items = [
        (
            name,
            str(args.base / name),
            str(args.primary / name),
            str(args.secondary / name),
            str(args.output / name),
            config,
        )
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(materialize_one, items), 1):
            print(f"[{index:03d}/{len(names):03d}] {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--gt", type=Path, required=True)
    for option in ("base", "primary", "secondary", "output"):
        screen_parser.add_argument(f"--{option}", type=Path, required=True)
    screen_parser.add_argument("--workers", type=int, default=8)
    screen_parser.set_defaults(function=screen)
    materialize_parser = subparsers.add_parser("materialize")
    for option in ("base", "primary", "secondary", "config", "output"):
        materialize_parser.add_argument(f"--{option}", type=Path, required=True)
    materialize_parser.add_argument("--workers", type=int, default=8)
    materialize_parser.set_defaults(function=materialize)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.function(args)


if __name__ == "__main__":
    main()
