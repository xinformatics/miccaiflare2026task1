#!/usr/bin/env python3
"""Materialize full-volume FreeTumor-free proxy masks without labels."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


DOCKER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCKER_ROOT.parents[2]
if str(DOCKER_ROOT) not in sys.path:
    sys.path.insert(0, str(DOCKER_ROOT))

from runtime_winner_mask_recipe import (  # noqa: E402
    confirm_with_stunet_p70_or_existing_support_proxy,
)


def validate_public_only_path(path: Path, role: str) -> None:
    """Fail closed if any input or output resolves into label/hidden storage."""
    normalized_parts = {
        part.casefold().replace("_", "-") for part in path.resolve().parts
    }
    forbidden_exact = {
        "hidden",
        "validation-hidden-images",
        "validation-hidden-labels",
        "validation-public-labels",
        "validation-public-label",
    }
    if normalized_parts & forbidden_exact:
        raise ValueError(f"{role} path is not public-image-only: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_binary(path: Path, reference: nib.Nifti1Image | None = None) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-4)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), (0, 1)).all():
        raise ValueError(f"nonbinary mask: {path}")
    return image, values.astype(bool, copy=False)


def materialize_case(item: tuple[str, dict[str, str]]) -> dict:
    name, rendered = item
    paths = {key: Path(value) for key, value in rendered.items()}
    reference, anchor = load_binary(paths["anchor"] / name)
    _, bbox = load_binary(paths["bbox"] / name, reference)
    _, whole = load_binary(paths["whole"] / name, reference)
    _, tumor = load_binary(paths["tumor"] / name, reference)
    probability_path = paths["stunet"] / f"{name.removesuffix('.nii.gz')}.prob16.npz"
    with np.load(probability_path, allow_pickle=False) as archive:
        probability = archive["lesion_probability_uint16"]
    result, stats = confirm_with_stunet_p70_or_existing_support_proxy(
        anchor=anchor,
        prompt_whole_body_lesion=whole,
        prompt_tumor=tumor,
        bbox_raw=bbox,
        stunet_probability_uint16=probability,
        spacing=tuple(float(value) for value in reference.header.get_zooms()[:3]),
    )
    if np.any(result & ~anchor):
        raise RuntimeError(f"proxy reference escaped anchor: {name}")
    destination = paths["output"] / name
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(result.astype(np.uint8), reference.affine, header), str(temporary))
    written = nib.load(str(temporary))
    if (
        written.shape != reference.shape
        or not np.allclose(written.affine, reference.affine, atol=1e-6)
        or not np.array_equal(np.asanyarray(written.dataobj).astype(bool), result)
    ):
        raise RuntimeError(f"proxy reference write validation failed: {name}")
    os.replace(temporary, destination)
    return {
        "case": name,
        "foreground_voxels": int(result.sum()),
        "anchor_components": stats.anchor_components,
        "stunet_confirmed_components": stats.stunet_confirmed_components,
        "proxy_confirmed_components": stats.proxy_confirmed_components,
        "rejected_components": stats.rejected_components,
    }


def inventory_sha256(directory: Path, names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0")
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--bbox", required=True, type=Path)
    parser.add_argument("--prompt-whole", required=True, type=Path)
    parser.add_argument("--prompt-tumor", required=True, type=Path)
    parser.add_argument("--stunet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dataset-role", choices=("public",), required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1,4]")
    for role, path in (
        ("anchor", args.anchor),
        ("bbox", args.bbox),
        ("prompt-whole", args.prompt_whole),
        ("prompt-tumor", args.prompt_tumor),
        ("stunet", args.stunet),
        ("output", args.output),
        ("terminal", args.terminal),
    ):
        validate_public_only_path(path, role)
    names = sorted(path.name for path in args.anchor.glob("*.nii.gz"))
    expected = set(names)
    if len(names) != 50:
        raise ValueError("proxy reference requires exactly 50 public cases")
    for role, directory in (
        ("bbox", args.bbox),
        ("whole", args.prompt_whole),
        ("tumor", args.prompt_tumor),
        ("stunet", args.stunet),
    ):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(f"{role} NIfTI inventory mismatch")
    probabilities = {
        f"{path.name.removesuffix('.prob16.npz')}.nii.gz"
        for path in args.stunet.glob("*.prob16.npz")
    }
    if probabilities != expected:
        raise ValueError("STU compact probability inventory mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = {
        "anchor": str(args.anchor.resolve()),
        "bbox": str(args.bbox.resolve()),
        "whole": str(args.prompt_whole.resolve()),
        "tumor": str(args.prompt_tumor.resolve()),
        "stunet": str(args.stunet.resolve()),
        "output": str(args.output.resolve()),
    }
    rows = []
    if args.workers == 1:
        for item in ((name, rendered) for name in names):
            rows.append(materialize_case(item))
            print(f"proxy reference progress {len(rows)}/50", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(materialize_case, (name, rendered)) for name in names]
            for future in as_completed(futures):
                rows.append(future.result())
                print(f"proxy reference progress {len(rows)}/50", flush=True)
    rows.sort(key=lambda row: row["case"])
    if {path.name for path in args.output.glob("*.nii.gz")} != expected:
        raise RuntimeError("proxy reference output inventory mismatch")
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "experiment": "full_volume_freetumor_free_existing_support_proxy_reference",
        "case_count": 50,
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
    print(json.dumps({key: payload[key] for key in ("status", "case_count", "output_inventory_sha256")}, indent=2))


if __name__ == "__main__":
    main()
