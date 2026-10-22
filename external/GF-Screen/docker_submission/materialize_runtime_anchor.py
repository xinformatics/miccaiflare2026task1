#!/usr/bin/env python3
"""Compose the pre-STU winner anchor from public-image-derived masks only."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from materialize_runtime_proxy_reference import (
    atomic_json,
    inventory_sha256,
    load_binary,
    sha256,
    validate_public_only_path,
)
from runtime_winner_mask_recipe import compose_runtime_anchor


DOCKER_ROOT = Path(__file__).resolve().parent


def materialize_case(item: tuple[str, dict[str, str]]) -> dict:
    name, rendered = item
    started = time.perf_counter()
    paths = {key: Path(value) for key, value in rendered.items()}
    reference, bbox = load_binary(paths["bbox"] / name)
    _, whole = load_binary(paths["whole"] / name, reference)
    _, tumor = load_binary(paths["tumor"] / name, reference)
    _, guided = load_binary(paths["guided"] / name, reference)
    _, cbg = load_binary(paths["cbg"] / name, reference)
    anchor, stats = compose_runtime_anchor(
        bbox_raw=bbox,
        prompt_whole_body_lesion=whole,
        prompt_tumor=tumor,
        guided_gf=guided,
        cbg_gf=cbg,
        spacing=tuple(float(value) for value in reference.header.get_zooms()[:3]),
    )
    destination = paths["output"] / name
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(anchor.astype(np.uint8), reference.affine, header), str(temporary))
    written = nib.load(str(temporary))
    if (
        written.shape != reference.shape
        or not np.allclose(written.affine, reference.affine, atol=1e-6)
        or not np.array_equal(np.asanyarray(written.dataobj).astype(bool), anchor)
    ):
        raise RuntimeError(f"runtime-anchor write validation failed: {name}")
    os.replace(temporary, destination)
    return {
        "case": name,
        "wall_seconds": time.perf_counter() - started,
        **{key: int(value) for key, value in vars(stats).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True, type=Path)
    parser.add_argument("--prompt-whole", required=True, type=Path)
    parser.add_argument("--prompt-tumor", required=True, type=Path)
    parser.add_argument("--guided", required=True, type=Path)
    parser.add_argument("--cbg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset-role", choices=("public",), required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    for role, path in (
        ("bbox", args.bbox),
        ("prompt-whole", args.prompt_whole),
        ("prompt-tumor", args.prompt_tumor),
        ("guided", args.guided),
        ("cbg", args.cbg),
        ("output", args.output),
        ("terminal", args.terminal),
    ):
        validate_public_only_path(path, role)
    names = sorted(path.name for path in args.bbox.glob("*.nii.gz"))
    expected = set(names)
    if len(names) != 50:
        raise ValueError("runtime-anchor materializer requires exactly 50 public cases")
    for role, directory in (
        ("whole", args.prompt_whole),
        ("tumor", args.prompt_tumor),
        ("guided", args.guided),
        ("cbg", args.cbg),
    ):
        if {path.name for path in directory.glob("*.nii.gz")} != expected:
            raise ValueError(f"{role} NIfTI inventory mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = {
        "bbox": str(args.bbox.resolve()),
        "whole": str(args.prompt_whole.resolve()),
        "tumor": str(args.prompt_tumor.resolve()),
        "guided": str(args.guided.resolve()),
        "cbg": str(args.cbg.resolve()),
        "output": str(args.output.resolve()),
    }
    wall_started = time.perf_counter()
    rows = []
    items = ((name, rendered) for name in names)
    if args.workers == 1:
        for item in items:
            rows.append(materialize_case(item))
            print(f"runtime anchor progress {len(rows)}/50", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(materialize_case, item) for item in items]
            for future in as_completed(futures):
                rows.append(future.result())
                print(f"runtime anchor progress {len(rows)}/50", flush=True)
    rows.sort(key=lambda row: row["case"])
    if {path.name for path in args.output.glob("*.nii.gz")} != expected:
        raise RuntimeError("runtime-anchor output inventory mismatch")
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "experiment": "pre_stunet_runtime_anchor",
        "case_count": 50,
        "wall_seconds": time.perf_counter() - wall_started,
        "cases": rows,
        "output": str(args.output.resolve()),
        "output_inventory_sha256": inventory_sha256(args.output, names),
        "runtime_recipe_sha256": sha256(DOCKER_ROOT / "runtime_winner_mask_recipe.py"),
        "dataset_role": "public",
        "public_images_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.terminal, payload)
    print(json.dumps({key: payload[key] for key in ("status", "case_count", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
