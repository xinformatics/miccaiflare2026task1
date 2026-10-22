#!/usr/bin/env python3
"""Preserve a base segmentation and union a frozen routed rescue expert."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--rescue", required=True, type=Path)
    parser.add_argument("--routing", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--role", required=True, choices=("public", "hidden"))
    args = parser.parse_args()

    base_names = {path.name for path in args.base.glob("*.nii.gz")}
    rescue_names = {path.name for path in args.rescue.glob("*.nii.gz")}
    if base_names != rescue_names or len(base_names) != args.expected:
        raise RuntimeError(
            f"expert inventory mismatch: base={len(base_names)} rescue={len(rescue_names)}"
        )
    with args.routing.open(newline="", encoding="utf-8") as stream:
        routing = {row["case"]: row for row in csv.DictReader(stream)}
    cases = {name.removesuffix(".nii.gz") for name in base_names}
    if set(routing) != cases:
        raise RuntimeError("routing inventory differs from prediction inventory")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing nonempty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    changed = []
    added_voxels = 0
    for case in sorted(cases):
        name = f"{case}.nii.gz"
        base_path = args.base / name
        if int(float(routing[case]["choose_f23"])) == 0:
            shutil.copyfile(base_path, args.output / name)
            continue
        base_image = nib.load(str(base_path))
        rescue_image = nib.load(str(args.rescue / name))
        if base_image.shape != rescue_image.shape or not np.allclose(
            base_image.affine, rescue_image.affine, rtol=0, atol=1e-4
        ):
            raise RuntimeError(f"geometry mismatch: {case}")
        base = np.asanyarray(base_image.dataobj) > 0
        rescue = np.asanyarray(rescue_image.dataobj) > 0
        union = base | rescue
        added_voxels += int(np.count_nonzero(union & ~base))
        changed.append(case)
        header = base_image.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(
            nib.Nifti1Image(union.astype(np.uint8), base_image.affine, header),
            str(args.output / name),
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "method": "base_union_routed_flare23",
        "role": args.role,
        "cases": len(cases),
        "changed_cases": changed,
        "added_voxels": added_voxels,
        "base": str(args.base.resolve()),
        "rescue": str(args.rescue.resolve()),
        "routing": {"path": str(args.routing.resolve()), "sha256": sha256(args.routing)},
        "output": str(args.output.resolve()),
        "labels_read": False,
        "uploaded": False,
    }
    atomic_json(args.output.parent / f"{args.output.name}.terminal.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
