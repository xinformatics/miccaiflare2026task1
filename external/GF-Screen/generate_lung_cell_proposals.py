#!/usr/bin/env python3
"""Generate source-separated, label-blind lung coverage cells from CT.

Unlike point/blob seeds, every occupied search cell is filled.  The eight
parity colours keep face/edge/corner-adjacent cells in separate proposal
sources, so connected-component extraction preserves one candidate per cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from generate_lung_blob_proposals import approximate_lung_region


COLOURS = 8


def atomic_nifti_temporary(destination: Path) -> Path:
    """Return a same-filesystem staging path outside the final-mask glob."""

    staging = destination.parent / ".atomic_temporary"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / f".{destination.name}.{os.getpid()}.nii.gz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def cell_colour(cell: np.ndarray) -> int:
    value = np.asarray(cell, dtype=np.int64)
    return int(((value[0] & 1) << 2) | ((value[1] & 1) << 1) | (value[2] & 1))


def native_cell_box(
    cell: np.ndarray,
    spacing: np.ndarray,
    stride_mm: float,
    shape: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    """Return a centred box whose physical extent never exceeds stride_mm."""

    cell = np.asarray(cell, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    shape_array = np.asarray(shape, dtype=np.int64)
    counts = np.maximum(np.floor(stride_mm / spacing).astype(np.int64), 1)
    counts = np.minimum(counts, shape_array)
    centre = np.rint(((cell + 0.5) * stride_mm / spacing) - 0.5).astype(np.int64)
    low = centre - counts // 2
    low = np.minimum(np.maximum(low, 0), shape_array - counts)
    high = low + counts
    return tuple(slice(int(a), int(b)) for a, b in zip(low, high))


def occupied_lung_cells(
    ct: np.ndarray,
    spacing: np.ndarray,
    target_spacing_mm: float,
    cell_stride_mm: float,
) -> list[np.ndarray]:
    zoom = np.asarray(spacing, dtype=np.float64) / float(target_spacing_mm)
    sampled = ndi.zoom(
        ct.astype(np.float32, copy=False),
        zoom,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    lung = approximate_lung_region(
        sampled, max(1, round(10.0 / float(target_spacing_mm)))
    )
    coordinates = np.argwhere(lung)
    if coordinates.size == 0:
        return []
    physical = coordinates * float(target_spacing_mm)
    cells = np.unique(
        np.floor(physical / float(cell_stride_mm)).astype(np.int32), axis=0
    )
    return [cell for cell in cells]


def paint_cells_by_colour(
    shape: tuple[int, int, int],
    spacing: np.ndarray,
    cells: list[np.ndarray],
    stride_mm: float,
) -> list[np.ndarray]:
    masks = [np.zeros(shape, dtype=bool) for _ in range(COLOURS)]
    for cell in cells:
        masks[cell_colour(cell)][native_cell_box(cell, spacing, stride_mm, shape)] = True
    return masks


def _run_case(task: dict[str, object]) -> dict[str, object]:
    image_path = Path(str(task["image"]))
    prediction_name = str(task["prediction_name"])
    output_root = Path(str(task["output_root"]))
    case_marker = output_root / "case_complete" / f"{prediction_name}.json"
    if bool(task.get("resume")) and case_marker.is_file():
        row = json.loads(case_marker.read_text(encoding="utf-8"))
        outputs = [output_root / f"colour{colour}" / prediction_name for colour in range(COLOURS)]
        if row.get("case") == prediction_name and all(path.is_file() for path in outputs):
            row["reused"] = True
            return row
    image = nib.load(str(image_path))
    ct = np.asanyarray(image.dataobj, dtype=np.float32)
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    stride_mm = float(task["cell_stride_mm"])
    cells = occupied_lung_cells(
        ct,
        spacing,
        float(task["target_spacing_mm"]),
        stride_mm,
    )
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    colour_counts = []
    # Emit one colour at a time. Full CTs can contain hundreds of millions of
    # voxels, so retaining eight dense masks simultaneously is unnecessary.
    for colour in range(COLOURS):
        mask = np.zeros(image.shape, dtype=bool)
        selected = [cell for cell in cells if cell_colour(cell) == colour]
        for cell in selected:
            mask[native_cell_box(cell, spacing, stride_mm, image.shape)] = True
        destination = output_root / f"colour{colour}" / prediction_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = atomic_nifti_temporary(destination)
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), image.affine, header),
            str(temporary),
        )
        os.replace(temporary, destination)
        colour_counts.append(len(selected))
        del mask
    maximum_extent = max(
        (
            max(
                (axis.stop - axis.start) * float(spacing[index])
                for index, axis in enumerate(native_cell_box(cell, spacing, stride_mm, image.shape))
            )
            for cell in cells
        ),
        default=0.0,
    )
    row = {
        "case": prediction_name,
        "cells": len(cells),
        "cells_by_colour": colour_counts,
        "maximum_component_axis_extent_mm": maximum_extent,
        "image_sha256": sha256_file(image_path),
        "reused": False,
    }
    case_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = case_marker.with_name(f".{case_marker.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, case_marker)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cohort")
    selection.add_argument("--all-cases", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-spacing-mm", type=float, default=2.5)
    parser.add_argument("--cell-stride-mm", type=float, default=48.0)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-complete-labels", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse a case only after all eight outputs and its atomic sidecar exist",
    )
    args = parser.parse_args()
    if min(args.target_spacing_mm, args.cell_stride_mm, args.workers) <= 0:
        raise ValueError("spacings and workers must be positive")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = (
        list(payload["items"])
        if args.all_cases
        else [
            item
            for item in payload["items"]
            if str(item.get("cohort")) == str(args.cohort)
        ]
    )
    incomplete = sum(item.get("complete_label") is not True for item in items)
    if len(items) != args.expected_count or (
        args.require_complete_labels and incomplete
    ):
        raise ValueError(
            "selection coverage/completeness mismatch: "
            f"{len(items)}/{args.expected_count}, incomplete={incomplete}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "image": item["image"],
            "prediction_name": item["prediction_name"],
            "output_root": str(args.output),
            "target_spacing_mm": args.target_spacing_mm,
            "cell_stride_mm": args.cell_stride_mm,
            "resume": args.resume,
        }
        for item in items
    ]
    if args.workers == 1:
        rows = [_run_case(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_run_case, tasks))
    expected = {str(item["prediction_name"]) for item in items}
    for colour in range(COLOURS):
        observed = {
            path.name for path in (args.output / f"colour{colour}").glob("*.nii.gz")
        }
        if observed != expected:
            raise ValueError(
                f"output coverage mismatch colour{colour}: {len(observed)}/{len(expected)}"
            )
    counts = [int(row["cells"]) for row in rows]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "filled_48mm_occupied_lung_cells_in_eight_parity_sources",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "cohort": "ALL" if args.all_cases else args.cohort,
        "cases": len(rows),
        "sources": COLOURS,
        "parameters": {
            "target_spacing_mm": args.target_spacing_mm,
            "cell_stride_mm": args.cell_stride_mm,
        },
        "cell_count": {
            "minimum": min(counts),
            "mean": float(np.mean(counts)),
            "maximum": max(counts),
        },
        "maximum_component_axis_extent_mm": max(
            float(row["maximum_component_axis_extent_mm"]) for row in rows
        ),
        "labels_read": False,
        "reused_cases": sum(bool(row.get("reused")) for row in rows),
        "complete_labels_required": bool(args.require_complete_labels),
        "public_or_hidden_data_read": False,
        "rows": rows,
    }
    marker = args.output / "PROPOSALS_COMPLETE.json"
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
