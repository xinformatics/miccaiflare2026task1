#!/usr/bin/env python3
"""Screen label-blind contour growth of an immutable anchor into model support.

Unlike morphological dilation, every added voxel must be foreground in an
independent support segmentation.  The finite-distance variants cannot create
a disconnected lesion; the touching-component variant imports only support
components that already overlap the anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage


from fast_official_lesion_metrics import score_case


CONFIGURATIONS = ("identity", "support_1mm", "support_2mm", "support_3mm", "support_5mm", "touching_component")


def transform(
    anchor: np.ndarray,
    support: np.ndarray,
    spacing: tuple[float, float, float],
    configuration: str,
    *,
    distance: np.ndarray | None = None,
    touching_support: np.ndarray | None = None,
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if anchor.shape != support.shape:
        raise ValueError("anchor/support shape mismatch")
    if configuration == "identity":
        return anchor.copy()
    if configuration == "touching_component":
        if touching_support is None:
            labels = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
            touching = np.unique(labels[anchor])
            keep = np.zeros(int(labels.max()) + 1, dtype=bool)
            keep[touching] = True
            keep[0] = False
            touching_support = keep[labels]
        result = anchor | touching_support
    elif configuration.startswith("support_") and configuration.endswith("mm"):
        radius = float(configuration.removeprefix("support_").removesuffix("mm"))
        if distance is None:
            distance = ndimage.distance_transform_edt(~anchor, sampling=spacing)
        result = anchor | (support & (distance <= radius))
    else:
        raise ValueError(f"unknown configuration: {configuration}")
    if not np.all(result[anchor]) or np.any((result & ~anchor) & ~support):
        raise AssertionError("anchor/support invariant failed")
    return result


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
    anchor, support, anchor_image, spacing = load_pair(Path(anchor_path), Path(support_path))
    target_image = nib.load(target_path)
    if anchor_image.shape != target_image.shape or not np.allclose(
        anchor_image.affine, target_image.affine, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"target geometry mismatch: {name}")
    target = np.asanyarray(target_image.dataobj).astype(bool)
    distance = ndimage.distance_transform_edt(~anchor, sampling=spacing)
    support_labels = cc3d.connected_components(support.astype(np.uint8), connectivity=26)
    touching = np.unique(support_labels[anchor])
    keep = np.zeros(int(support_labels.max()) + 1, dtype=bool)
    keep[touching] = True
    keep[0] = False
    touching_support = keep[support_labels]
    rows = []
    for configuration in CONFIGURATIONS:
        prediction = transform(
            anchor,
            support,
            spacing,
            configuration,
            distance=distance,
            touching_support=touching_support,
        )
        dsc, nsd = score_case(target.astype(np.uint8), prediction.astype(np.uint8), spacing)
        rows.append(
            (configuration, float(dsc), float(nsd), int((prediction & ~anchor).sum()))
        )
    return name.removesuffix(".nii.gz"), rows


def materialize_one(item: tuple[str, str, str], configuration: str, output: Path) -> str:
    name, anchor_path, support_path = item
    anchor, support, reference, spacing = load_pair(Path(anchor_path), Path(support_path))
    prediction = transform(anchor, support, spacing, configuration)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = output / f".{name}.{os.getpid()}.tmp.nii.gz"
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), reference.affine, header), str(temporary))
    os.replace(temporary, output / name)
    return name


def inventory(anchor: Path, support: Path) -> list[str]:
    names = sorted(path.name for path in anchor.glob("*.nii.gz"))
    if not names or names != sorted(path.name for path in support.glob("*.nii.gz")):
        raise ValueError("anchor/support inventory mismatch")
    return names


def screen(args) -> None:
    names = inventory(args.anchor, args.support)
    targets = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if names != targets:
        raise ValueError("prediction/label inventory mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    by_configuration = {name: [] for name in CONFIGURATIONS}
    items = [
        (name, str(args.anchor / name), str(args.support / name), str(args.labels / name))
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, rows) in enumerate(executor.map(score_one, items), 1):
            for configuration, dsc, nsd, added in rows:
                by_configuration[configuration].append((case, dsc, nsd, added))
            print(f"[{index:03d}/{len(names):03d}] {case}", flush=True)
    summary = []
    for configuration, rows in by_configuration.items():
        metrics = args.output / f"{configuration}.metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD", "added_voxels"))
            writer.writerows(rows)
        dsc = float(np.mean([row[1] for row in rows]))
        nsd = float(np.mean([row[2] for row in rows]))
        summary.append(
            {
                "configuration": configuration,
                "DSC": dsc,
                "NSD": nsd,
                "combined": (dsc + nsd) / 2,
                "changed_cases": sum(row[3] > 0 for row in rows),
                "added_voxels": sum(row[3] for row in rows),
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
                "operation": "immutable anchor growth constrained to independent support",
                "anchor": str(args.anchor.resolve()),
                "support": str(args.support.resolve()),
                "cases": len(names),
                "configurations": CONFIGURATIONS,
                "labels_used_for_screen_only": True,
                "hidden_read": False,
                "submission_created": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((args.output / "summary.tsv").read_text(encoding="utf-8"), end="")


def materialize(args) -> None:
    names = inventory(args.anchor, args.support)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output must be empty: {args.output}")
    items = [(name, str(args.anchor / name), str(args.support / name)) for name in names]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(materialize_one, item, args.configuration, args.output)
            for item in items
        ]
        for index, future in enumerate(futures, 1):
            print(f"[{index:03d}/{len(names):03d}] {future.result()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen_parser = subparsers.add_parser("screen")
    screen_parser.add_argument("--anchor", type=Path, required=True)
    screen_parser.add_argument("--support", type=Path, required=True)
    screen_parser.add_argument("--labels", type=Path, required=True)
    screen_parser.add_argument("--output", type=Path, required=True)
    screen_parser.add_argument("--workers", type=int, default=2)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--anchor", type=Path, required=True)
    materialize_parser.add_argument("--support", type=Path, required=True)
    materialize_parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    materialize_parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.command == "screen":
        screen(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
