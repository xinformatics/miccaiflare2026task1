#!/usr/bin/env python3
"""Compose arbitrary-inventory deployment masks from learned stage products."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from runtime_winner_mask_recipe import compose_runtime_anchor, existing_support_proxy_mask
from winner_mask_recipe import filter_components_physical_6


def prediction_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = path.stem
    if name.endswith("_0000"):
        name = name[:-5]
    return f"{name}.nii.gz"


def binary(path: Path, reference: nib.Nifti1Image) -> np.ndarray:
    image = nib.load(str(path))
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-6):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"non-binary mask: {path}")
    return values.astype(bool, copy=False)


def save(mask: np.ndarray, reference: nib.Nifti1Image, destination: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header), str(temporary))
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--bbox", required=True, type=Path)
    parser.add_argument("--whole", type=Path)
    parser.add_argument("--tumor", type=Path)
    parser.add_argument("--guided", type=Path)
    parser.add_argument("--cbg", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("gf", "anchor", "proxy"), required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    sources = sorted(args.images.glob("*.nii.gz"))
    if not sources:
        raise FileNotFoundError(f"no NIfTI inputs in {args.images}")
    if args.mode != "gf" and any(value is None for value in (args.whole, args.tumor, args.guided, args.cbg)):
        raise ValueError("anchor/proxy modes require whole, tumor, guided, and CBG masks")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    started_all = time.perf_counter()
    for source in sources:
        started = time.perf_counter()
        name = prediction_name(source)
        reference = nib.load(str(source))
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        bbox = binary(args.bbox / name, reference)
        if args.mode == "gf":
            result = filter_components_physical_6(bbox, spacing, 1000.0)
            stats = {"result_voxels": int(result.sum())}
        else:
            whole = binary(args.whole / name, reference)
            tumor = binary(args.tumor / name, reference)
            guided = binary(args.guided / name, reference)
            cbg = binary(args.cbg / name, reference)
            anchor, anchor_stats = compose_runtime_anchor(
                bbox_raw=bbox,
                prompt_whole_body_lesion=whole,
                prompt_tumor=tumor,
                guided_gf=guided,
                cbg_gf=cbg,
                spacing=spacing,
            )
            result = anchor
            if args.mode == "proxy":
                result = existing_support_proxy_mask(
                    anchor=anchor,
                    prompt_whole_body_lesion=whole,
                    prompt_tumor=tumor,
                    bbox_raw=bbox,
                    spacing=spacing,
                )
            stats = {**vars(anchor_stats), "result_voxels": int(result.sum())}
        save(result, reference, args.output / name)
        rows.append({"case": name, "seconds": time.perf_counter() - started, **stats})
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    payload = {
        "status": "COMPLETE",
        "mode": args.mode,
        "case_count": len(rows),
        "wall_seconds": time.perf_counter() - started_all,
        "cases": rows,
    }
    args.terminal.parent.mkdir(parents=True, exist_ok=True)
    args.terminal.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
