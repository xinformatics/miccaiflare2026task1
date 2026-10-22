#!/usr/bin/env python3
"""Screen conservative risk gates for support components touching an anchor.

The deployed ``touching_component`` operation accepts an entire independent
support component after even one overlapping voxel.  This module preserves the
anchor and whole-component recovery, but can reject contacts that are too small
or extensions that are implausibly large relative to their anchor overlap.
Only inference-observable geometry is used by :func:`transform`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from fast_official_lesion_metrics import score_case


@dataclass(frozen=True)
class RiskGate:
    minimum_overlap_mm3: float = 0.0
    minimum_overlap_fraction: float = 0.0
    maximum_extension_ratio: float = float("inf")
    maximum_added_mm3: float = float("inf")


# A compact, predeclared physics-based grid.  The baseline is included as an
# identity check and ``touch_all`` must reproduce the submitted implementation.
CONFIGURATIONS: dict[str, RiskGate | None] = {
    "identity": None,
    "touch_all": RiskGate(),
    "extension_ratio_0p25": RiskGate(maximum_extension_ratio=0.25),
    "extension_ratio_0p35": RiskGate(maximum_extension_ratio=0.35),
    "extension_ratio_0p50": RiskGate(maximum_extension_ratio=0.50),
    "extension_ratio_0p75": RiskGate(maximum_extension_ratio=0.75),
    "extension_ratio_1p00": RiskGate(maximum_extension_ratio=1.00),
    "extension_ratio_1p50": RiskGate(maximum_extension_ratio=1.50),
    "extension_ratio_2p00": RiskGate(maximum_extension_ratio=2.00),
    "extension_ratio_3p00": RiskGate(maximum_extension_ratio=3.00),
    "extension_ratio_4p00": RiskGate(maximum_extension_ratio=4.00),
    "added_mm3_2500": RiskGate(maximum_added_mm3=2500.0),
    "added_mm3_5000": RiskGate(maximum_added_mm3=5000.0),
    "added_mm3_10000": RiskGate(maximum_added_mm3=10000.0),
    "added_mm3_25000": RiskGate(maximum_added_mm3=25000.0),
    "added25k_ratio3": RiskGate(maximum_extension_ratio=3.0, maximum_added_mm3=25000.0),
}


def transform(
    anchor: np.ndarray,
    support: np.ndarray,
    voxel_mm3: float,
    configuration: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return an immutable-anchor prediction and component audit counts."""

    anchor = np.asarray(anchor, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if anchor.shape != support.shape:
        raise ValueError("anchor/support shape mismatch")
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("voxel volume must be finite and positive")
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {configuration}")
    gate = CONFIGURATIONS[configuration]
    if gate is None:
        return anchor.copy(), {
            "touching_components": 0,
            "accepted_components": 0,
            "added_voxels": 0,
        }

    labels = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
    return transform_precomputed(anchor, support, voxel_mm3, configuration, labels)


def transform_precomputed(
    anchor: np.ndarray,
    support: np.ndarray,
    voxel_mm3: float,
    configuration: str,
    labels: np.ndarray,
    statistics: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply one gate using a support-component map shared across a case grid."""

    anchor = np.asarray(anchor, dtype=bool)
    support = np.asarray(support, dtype=bool)
    labels = np.asarray(labels)
    if labels.shape != anchor.shape or support.shape != anchor.shape:
        raise ValueError("precomputed component geometry shape mismatch")
    gate = CONFIGURATIONS[configuration]
    if gate is None:
        return anchor.copy(), {
            "touching_components": 0,
            "accepted_components": 0,
            "added_voxels": 0,
        }
    component_count = int(labels.max())
    if statistics is None:
        sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
        overlap = np.bincount(labels[anchor], minlength=component_count + 1)
    else:
        sizes, overlap = statistics
        if sizes.shape != (component_count + 1,) or overlap.shape != sizes.shape:
            raise ValueError("precomputed component statistics shape mismatch")
    added = sizes - overlap
    touching = overlap > 0
    touching[0] = False
    overlap_mm3 = overlap.astype(np.float64) * voxel_mm3
    overlap_fraction = np.divide(
        overlap,
        sizes,
        out=np.zeros_like(overlap, dtype=np.float64),
        where=sizes > 0,
    )
    extension_ratio = np.divide(
        added,
        overlap,
        out=np.full_like(added, np.inf, dtype=np.float64),
        where=overlap > 0,
    )
    added_mm3 = added.astype(np.float64) * voxel_mm3
    accepted = (
        touching
        & (overlap_mm3 >= gate.minimum_overlap_mm3)
        & (overlap_fraction >= gate.minimum_overlap_fraction)
        & (extension_ratio <= gate.maximum_extension_ratio)
        & (added_mm3 <= gate.maximum_added_mm3)
    )
    accepted[0] = False
    imported = accepted[labels]
    result = anchor | imported
    if not np.all(result[anchor]) or np.any((result & ~anchor) & ~support):
        raise AssertionError("immutable anchor/support invariant failed")
    return result, {
        "touching_components": int(touching.sum()),
        "accepted_components": int(accepted.sum()),
        "added_voxels": int((result & ~anchor).sum()),
    }


def load_pair(anchor_path: Path, support_path: Path):
    anchor_image = nib.load(str(anchor_path))
    support_image = nib.load(str(support_path))
    if anchor_image.shape != support_image.shape or not np.allclose(
        anchor_image.affine, support_image.affine, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {anchor_path.name}")
    anchor = np.asanyarray(anchor_image.dataobj).astype(bool)
    support = np.asanyarray(support_image.dataobj).astype(bool)
    spacing = tuple(float(value) for value in anchor_image.header.get_zooms()[:3])
    return anchor, support, anchor_image, spacing


def score_one(item: tuple[str, str, str, str]):
    name, anchor_path, support_path, target_path = item
    anchor, support, reference, spacing = load_pair(Path(anchor_path), Path(support_path))
    target_image = nib.load(target_path)
    if reference.shape != target_image.shape or not np.allclose(
        reference.affine, target_image.affine, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"target geometry mismatch: {name}")
    target = np.asanyarray(target_image.dataobj).astype(bool)
    voxel_mm3 = float(np.prod(spacing))
    support_labels = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
    component_count = int(support_labels.max())
    statistics = (
        np.bincount(support_labels.ravel(), minlength=component_count + 1),
        np.bincount(support_labels[anchor], minlength=component_count + 1),
    )
    rows = []
    for configuration in CONFIGURATIONS:
        prediction, audit = transform_precomputed(
            anchor, support, voxel_mm3, configuration, support_labels, statistics
        )
        dsc, nsd = score_case(target.astype(np.uint8), prediction.astype(np.uint8), spacing)
        rows.append((configuration, float(dsc), float(nsd), audit))
    return name.removesuffix(".nii.gz"), rows


def inventory(anchor: Path, support: Path) -> list[str]:
    names = sorted(path.name for path in anchor.glob("*.nii.gz"))
    if not names or names != sorted(path.name for path in support.glob("*.nii.gz")):
        raise ValueError("anchor/support inventory mismatch")
    return names


def screen(args: argparse.Namespace) -> None:
    names = inventory(args.anchor, args.support)
    if names != sorted(path.name for path in args.labels.glob("*.nii.gz")):
        raise ValueError("prediction/label inventory mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    by_configuration = {name: [] for name in CONFIGURATIONS}
    tasks = [
        (name, str(args.anchor / name), str(args.support / name), str(args.labels / name))
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, rows) in enumerate(executor.map(score_one, tasks), 1):
            for configuration, dsc, nsd, audit in rows:
                by_configuration[configuration].append((case, dsc, nsd, audit))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)

    summary = []
    for configuration, rows in by_configuration.items():
        metrics = args.output / f"{configuration}.metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("Name", "Lesion_DSC", "Lesion_NSD", "touching_components", "accepted_components", "added_voxels")
            )
            for case, dsc, nsd, audit in rows:
                writer.writerow((case, dsc, nsd, audit["touching_components"], audit["accepted_components"], audit["added_voxels"]))
        summary.append(
            {
                "configuration": configuration,
                "DSC": float(np.mean([row[1] for row in rows])),
                "NSD": float(np.mean([row[2] for row in rows])),
                "accepted_components": sum(row[3]["accepted_components"] for row in rows),
                "added_voxels": sum(row[3]["added_voxels"] for row in rows),
                "metrics": str(metrics.resolve()),
            }
        )
    summary.sort(key=lambda row: (-row["DSC"], -row["NSD"], row["configuration"]))
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "whole touching support component with inference-observable contact-risk gates",
                "anchor": str(args.anchor.resolve()),
                "support": str(args.support.resolve()),
                "cases": len(names),
                "configurations": {name: None if gate is None else gate.__dict__ for name, gate in CONFIGURATIONS.items()},
                "labels_used_for_fixed_public_screen_only": True,
                "hidden_read": False,
                "submission_created": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.tsv").read_text(encoding="utf-8"), end="")


def materialize_one(item: tuple[str, str, str, str]) -> str:
    name, anchor_path, support_path, output_path = item
    anchor, support, reference, spacing = load_pair(Path(anchor_path), Path(support_path))
    configuration = Path(output_path).parent.name
    prediction, _ = transform(anchor, support, float(np.prod(spacing)), configuration)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = Path(output_path)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header), str(temporary))
    os.replace(temporary, output)
    return name


def materialize(args: argparse.Namespace) -> None:
    names = inventory(args.anchor, args.support)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output must be empty: {args.output}")
    if args.output.name != args.configuration:
        raise ValueError("output directory name must equal the configuration")
    tasks = [(name, str(args.anchor / name), str(args.support / name), str(args.output / name)) for name in names]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(materialize_one, tasks), 1):
            print(f"[{index:03d}/{len(names):03d}] {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--anchor", type=Path, required=True)
    common.add_argument("--support", type=Path, required=True)
    common.add_argument("--workers", type=int, default=2)
    screen_parser = subparsers.add_parser("screen", parents=[common])
    screen_parser.add_argument("--labels", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize", parents=[common])
    materialize_parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "screen":
        screen(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
