#!/usr/bin/env python3
"""Add independently confirmed disconnected components to an immutable anchor.

The operation is label-blind. Every anchor voxel is preserved exactly, and a
candidate connected component is added whole only when it is disconnected from
the anchor, satisfies physical-volume bounds, and receives the required number
of independent support votes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj).astype(bool), image


def validate_geometry(
    reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str
) -> None:
    if reference.shape != other.shape or not np.allclose(
        reference.affine, other.affine, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"geometry mismatch: {name}")


def materialize_case(
    base_path: Path,
    candidate_path: Path,
    support_paths: list[Path],
    output_path: Path,
    *,
    min_mm3: float,
    max_mm3: float,
    min_support_fraction: float,
    min_support_mm3: float,
    min_support_models: int,
) -> list[dict[str, object]]:
    base, base_image = load_mask(base_path)
    candidate, candidate_image = load_mask(candidate_path)
    validate_geometry(base_image, candidate_image, f"candidate/{base_path.name}")
    supports = []
    for path in support_paths:
        support, image = load_mask(path)
        validate_geometry(base_image, image, f"support/{path.parent.name}/{base_path.name}")
        supports.append(support)

    labels = cc3d.connected_components(candidate.astype(np.uint8), connectivity=26)
    counts = np.bincount(labels.ravel())
    voxel_mm3 = float(abs(np.linalg.det(base_image.affine[:3, :3])))
    if not math.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError(f"invalid voxel volume: {base_path}")
    volumes = counts.astype(np.float64) * voxel_mm3
    base_overlap = np.bincount(labels[base], minlength=len(counts))
    votes = np.zeros(len(counts), dtype=np.int16)
    maximum_support_fraction = np.zeros(len(counts), dtype=np.float64)
    maximum_support_mm3 = np.zeros(len(counts), dtype=np.float64)
    for support in supports:
        overlap = np.bincount(labels[support], minlength=len(counts))
        fraction = np.divide(
            overlap,
            counts,
            out=np.zeros(len(counts), dtype=np.float64),
            where=counts > 0,
        )
        overlap_mm3 = overlap.astype(np.float64) * voxel_mm3
        votes += (
            (fraction >= min_support_fraction) & (overlap_mm3 >= min_support_mm3)
        ).astype(np.int16)
        maximum_support_fraction = np.maximum(maximum_support_fraction, fraction)
        maximum_support_mm3 = np.maximum(maximum_support_mm3, overlap_mm3)

    eligible = base_overlap == 0
    eligible &= volumes >= min_mm3
    eligible &= volumes <= max_mm3
    eligible &= votes >= min_support_models
    eligible[0] = False
    addition = eligible[labels]
    output = base | addition
    if not np.all(output[base]):
        raise AssertionError("immutable-anchor invariant failed")
    if np.any(addition & ~candidate):
        raise AssertionError("addition escaped candidate support")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = base_image.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp.nii.gz"
    )
    nib.save(
        nib.Nifti1Image(output.astype(np.uint8), base_image.affine, header),
        str(temporary),
    )
    os.replace(temporary, output_path)

    rows = []
    for component_id in range(1, len(counts)):
        rows.append(
            {
                "case": base_path.name.removesuffix(".nii.gz"),
                "component_id": component_id,
                "candidate_mm3": float(volumes[component_id]),
                "overlaps_anchor": int(base_overlap[component_id] > 0),
                "support_votes": int(votes[component_id]),
                "max_support_fraction": float(maximum_support_fraction[component_id]),
                "max_support_mm3": float(maximum_support_mm3[component_id]),
                "accepted": int(eligible[component_id]),
            }
        )
    return rows


def common_names(directory: Path) -> list[str]:
    names = sorted(path.name for path in directory.glob("*.nii.gz"))
    if not names:
        raise FileNotFoundError(f"no NIfTI masks in {directory}")
    return names


def validate_inventory(names: list[str], directory: Path) -> None:
    observed = common_names(directory)
    if observed != names:
        expected_set, observed_set = set(names), set(observed)
        raise ValueError(
            f"case inventory mismatch in {directory}: "
            f"missing={sorted(expected_set-observed_set)[:3]} "
            f"extra={sorted(observed_set-expected_set)[:3]}"
        )


def _worker(item: tuple[object, ...]) -> list[dict[str, object]]:
    (
        base,
        candidate,
        supports,
        output,
        min_mm3,
        max_mm3,
        min_support_fraction,
        min_support_mm3,
        min_support_models,
    ) = item
    return materialize_case(
        Path(base),
        Path(candidate),
        [Path(path) for path in supports],
        Path(output),
        min_mm3=float(min_mm3),
        max_mm3=float(max_mm3),
        min_support_fraction=float(min_support_fraction),
        min_support_mm3=float(min_support_mm3),
        min_support_models=int(min_support_models),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--support", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--min-mm3", type=float, default=1000.0)
    parser.add_argument("--max-mm3", type=float, default=math.inf)
    parser.add_argument("--min-support-fraction", type=float, default=0.10)
    parser.add_argument("--min-support-mm3", type=float, default=100.0)
    parser.add_argument("--min-support-models", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    paths = [args.base, args.candidate, *args.support, args.output]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved[:-1])) != len(resolved[:-1]):
        raise ValueError("base, candidate, and support directories must be independent")
    if resolved[-1] in set(resolved[:-1]):
        raise ValueError("output must not overwrite an input directory")
    if (
        args.expected_count <= 0
        or args.workers <= 0
        or args.min_mm3 < 0
        or args.max_mm3 < args.min_mm3
        or not 0 <= args.min_support_fraction <= 1
        or args.min_support_mm3 < 0
        or not 1 <= args.min_support_models <= len(args.support)
    ):
        raise ValueError("invalid immutable-addition configuration")

    names = common_names(args.base)
    if len(names) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} anchor cases, found {len(names)}"
        )
    for directory in (args.candidate, *args.support):
        validate_inventory(names, directory)
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError(f"output directory must be empty: {args.output}")

    items = [
        (
            str(args.base / name),
            str(args.candidate / name),
            [str(directory / name) for directory in args.support],
            str(args.output / name),
            args.min_mm3,
            args.max_mm3,
            args.min_support_fraction,
            args.min_support_mm3,
            args.min_support_models,
        )
        for name in names
    ]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, case_rows in enumerate(executor.map(_worker, items), 1):
            rows.extend(case_rows)
            print(f"[{index:03d}/{len(items):03d}] {names[index-1]}", flush=True)

    fieldnames = [
        "case",
        "component_id",
        "candidate_mm3",
        "overlaps_anchor",
        "support_votes",
        "max_support_fraction",
        "max_support_mm3",
        "accepted",
    ]
    component_csv = args.audit.with_suffix(".components.csv")
    component_csv.parent.mkdir(parents=True, exist_ok=True)
    with component_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    output_names = common_names(args.output)
    if output_names != names:
        raise AssertionError("output coverage mismatch")
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "operation": "immutable whole-component addition",
        "base": str(args.base.resolve()),
        "candidate": str(args.candidate.resolve()),
        "support": [str(path.resolve()) for path in args.support],
        "output": str(args.output.resolve()),
        "cases": len(names),
        "candidate_components": len(rows),
        "accepted_components": sum(int(row["accepted"]) for row in rows),
        "accepted_mm3": float(
            sum(float(row["candidate_mm3"]) for row in rows if row["accepted"])
        ),
        "configuration": {
            "connectivity": 26,
            "min_mm3": args.min_mm3,
            "max_mm3": args.max_mm3,
            "min_support_fraction": args.min_support_fraction,
            "min_support_mm3": args.min_support_mm3,
            "min_support_models": args.min_support_models,
        },
        "components": str(component_csv.resolve()),
        "components_sha256": sha256(component_csv),
        "labels_read": False,
        "anchor_preserved": True,
    }
    rendered = json.dumps(payload, indent=2)
    temporary = args.audit.with_name(f".{args.audit.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered + "\n", encoding="utf-8")
    os.replace(temporary, args.audit)
    print(rendered)


if __name__ == "__main__":
    main()
