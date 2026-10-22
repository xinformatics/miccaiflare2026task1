#!/usr/bin/env python3
"""Create label-free top-K connected-component prompts from a frozen anchor."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


NIFTI_SUFFIX = ".nii.gz"


def voxel_volume_mm3(image: nib.Nifti1Image) -> float:
    value = float(abs(np.linalg.det(np.asarray(image.affine, dtype=np.float64)[:3, :3])))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("invalid NIfTI voxel volume")
    return value


def select_components(mask: np.ndarray, voxel_mm3: float, top_k: int, min_mm3: float, max_mm3: float) -> tuple[np.ndarray, list[dict[str, object]]]:
    if mask.ndim != 3:
        raise ValueError("component prompt mask must be 3-D")
    labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    eligible = [
        (label_id, int(sizes[label_id]), float(sizes[label_id] * voxel_mm3))
        for label_id in range(1, count + 1)
        if min_mm3 <= float(sizes[label_id] * voxel_mm3) <= max_mm3
    ]
    eligible.sort(key=lambda row: (-row[1], row[0]))
    chosen = eligible[:top_k]
    chosen_ids = {row[0] for row in chosen}
    selected = np.isin(labels, list(chosen_ids)) if chosen_ids else np.zeros(mask.shape, dtype=bool)
    records = [
        {"rank": rank, "label_id": label_id, "voxels": voxels, "volume_mm3": volume}
        for rank, (label_id, voxels, volume) in enumerate(chosen, 1)
    ]
    return selected, records


def save_binary_like(image: nib.Nifti1Image, mask: np.ndarray, output: Path) -> None:
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(mask.astype(np.uint8), image.affine, header)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp{NIFTI_SUFFIX}")
    nib.save(result, temporary)
    os.replace(temporary, output)


def prepare(input_dir: Path, output_dir: Path, top_k: int, min_mm3: float, max_mm3: float, expected_count: int | None) -> dict[str, object]:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input and output prompt directories must differ")
    files = sorted(input_dir.glob(f"*{NIFTI_SUFFIX}"))
    if expected_count is not None and len(files) != expected_count:
        raise ValueError(f"expected {expected_count} anchor masks, found {len(files)}")
    if not files:
        raise ValueError("no anchor masks found")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    selected_total = 0
    for path in files:
        image = nib.load(path)
        mask = np.asarray(image.dataobj) > 0
        selected, components = select_components(
            mask, voxel_volume_mm3(image), top_k, min_mm3, max_mm3
        )
        save_binary_like(image, selected, output_dir / path.name)
        selected_total += len(components)
        rows.append(
            {
                "case": path.name[: -len(NIFTI_SUFFIX)],
                "anchor_voxels": int(mask.sum()),
                "prompt_voxels": int(selected.sum()),
                "selected_components": len(components),
                "components": components,
            }
        )
    observed = {path.name for path in output_dir.glob(f"*{NIFTI_SUFFIX}")}
    expected = {path.name for path in files}
    if observed != expected:
        raise ValueError("top-K prompt output inventory mismatch")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "largest physical connected components of frozen anchor; label-free",
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
        "cases": len(files),
        "top_k": top_k,
        "min_mm3": min_mm3,
        "max_mm3": max_mm3,
        "selected_components": selected_total,
        "records": rows,
        "labels_read": False,
        "hidden_data_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-mm3", type=float, default=0.0)
    parser.add_argument("--max-mm3", type=float, default=float("inf"))
    parser.add_argument("--expected-count", type=int)
    args = parser.parse_args()
    if args.top_k < 1 or args.min_mm3 < 0 or args.max_mm3 < args.min_mm3:
        raise ValueError("invalid top-K component bounds")
    payload = prepare(
        args.input,
        args.output,
        args.top_k,
        args.min_mm3,
        args.max_mm3,
        args.expected_count,
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit.with_name(f".{args.audit.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.audit)
    csv_path = args.audit.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("case", "anchor_voxels", "prompt_voxels", "selected_components"),
        )
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in writer.fieldnames} for row in payload["records"]
        )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
