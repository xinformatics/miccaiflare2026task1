#!/usr/bin/env python3
"""Apply one frozen component-wise boundary transform to unlabeled masks."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from screen_boundary_postprocess import CONFIGS, componentwise_transform


def materialize_one(item: tuple[str, str, str]) -> tuple[str, int, int]:
    source_name, output_name, configuration = item
    source = Path(source_name)
    output = Path(output_name)
    image = nib.load(str(source))
    mask = np.asanyarray(image.dataobj).astype(bool)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    transformed = componentwise_transform(mask, spacing, configuration)
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(
        nib.Nifti1Image(transformed.astype(np.uint8), image.affine, header),
        str(temporary),
    )
    os.replace(temporary, output)
    return source.name, int(mask.sum()), int(transformed.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--configuration", choices=CONFIGS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    sources = sorted(args.predictions.glob("*.nii.gz"))
    if len(sources) != args.expected_count:
        parser.error(
            f"expected {args.expected_count} predictions, found {len(sources)}"
        )
    if args.workers < 1:
        parser.error("workers must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    expected_names = {source.name for source in sources}
    extras = {
        path.name for path in args.output.glob("*.nii.gz") if path.name not in expected_names
    }
    if extras:
        raise RuntimeError(f"unexpected existing outputs: {sorted(extras)[:3]}")
    items = [
        (str(source), str(args.output / source.name), args.configuration)
        for source in sources
    ]
    changed = 0
    foreground_delta = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (name, before, after) in enumerate(executor.map(materialize_one, items), 1):
            changed += int(before != after)
            foreground_delta += after - before
            print(
                f"[{index:03d}/{len(items):03d}] {name}: {before:,} -> {after:,}",
                flush=True,
            )
    observed = {path.name for path in args.output.glob("*.nii.gz")}
    if observed != expected_names:
        raise RuntimeError("materialized output coverage mismatch")
    print(
        f"complete configuration={args.configuration} cases={len(items)} "
        f"changed_cases={changed} foreground_delta={foreground_delta:+d}"
    )


if __name__ == "__main__":
    main()
