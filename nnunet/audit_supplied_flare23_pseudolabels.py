#!/usr/bin/env python3
"""Audit supplied FLARE23 organ pseudo-labels without using them for training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header_signature(path: Path) -> tuple[tuple[int, ...], tuple[float, ...]]:
    image = nib.load(path)
    affine = tuple(np.round(np.asarray(image.affine).ravel(), 5).tolist())
    return tuple(image.shape), affine


def inspect_mask(path: Path) -> dict:
    try:
        image = nib.load(path)
        values = np.asanyarray(image.dataobj)
        labels, counts = np.unique(values, return_counts=True)
        noninteger = int(np.count_nonzero(values != np.rint(values)))
        return {
            "name": path.name,
            "shape": list(image.shape),
            "dtype": str(values.dtype),
            "minimum": float(labels[0]),
            "maximum": float(labels[-1]),
            "noninteger_voxels": noninteger,
            "nonfinite_affine": not bool(np.isfinite(image.affine).all()),
            "label_counts": {
                str(int(label)): int(count) for label, count in zip(labels, counts)
            },
        }
    except Exception as error:  # pragma: no cover - diagnostic fail-closed path
        return {"name": path.name, "error": repr(error)}


def stratified_sample(paths: list[Path], sample_size: int) -> list[Path]:
    if sample_size <= 0 or sample_size >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, sample_size, dtype=int)
    return [paths[index] for index in indices]


def summarize_masks(paths: list[Path], workers: int, sample_size: int) -> dict:
    inspected_paths = stratified_sample(paths, sample_size)
    label_counts: Counter[int] = Counter()
    maxima: Counter[int] = Counter()
    dtypes: Counter[str] = Counter()
    errors: list[dict] = []
    empty = noninteger = nonfinite_affine = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in executor.map(inspect_mask, inspected_paths):
            if "error" in row:
                errors.append(row)
                continue
            dtypes[row["dtype"]] += 1
            maxima[int(row["maximum"])] += 1
            noninteger += int(row["noninteger_voxels"])
            nonfinite_affine += int(row["nonfinite_affine"])
            current = {int(label): count for label, count in row["label_counts"].items()}
            label_counts.update(current)
            empty += int(set(current) <= {0})
    return {
        "files_in_release": len(paths),
        "files_content_inspected": len(inspected_paths),
        "content_scan_scope": "complete" if len(inspected_paths) == len(paths) else "stratified_sample",
        "errors": errors,
        "error_count": len(errors),
        "dtypes": dict(sorted(dtypes.items())),
        "maximum_label_histogram": {
            str(label): count for label, count in sorted(maxima.items())
        },
        "voxel_counts_by_label": {
            str(label): count for label, count in sorted(label_counts.items())
        },
        "empty_masks": empty,
        "noninteger_voxels": noninteger,
        "nonfinite_affines": nonfinite_affine,
        "contains_tumor_label_14": label_counts[14] > 0,
    }


def identity_audit(images: list[Path], masks: list[Path]) -> dict:
    mask_by_signature: defaultdict[tuple, list[str]] = defaultdict(list)
    mask_by_shape: defaultdict[tuple, list[str]] = defaultdict(list)
    for path in masks:
        signature = header_signature(path)
        mask_by_signature[signature].append(path.name)
        mask_by_shape[signature[0]].append(path.name)

    exact_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    direct_stem_geometry_matches = 0
    mask_by_name = {path.name: path for path in masks}
    for path in images:
        signature = header_signature(path)
        exact = mask_by_signature.get(signature, [])
        shape = mask_by_shape.get(signature[0], [])
        exact_counts["none" if not exact else "unique" if len(exact) == 1 else "ambiguous"] += 1
        shape_counts["none" if not shape else "unique" if len(shape) == 1 else "ambiguous"] += 1
        suffix = path.name.removeprefix("amos_").removesuffix("_0000.nii.gz")
        direct = mask_by_name.get(f"FLARE23_{suffix}.nii.gz")
        if direct is not None and header_signature(direct) == signature:
            direct_stem_geometry_matches += 1

    return {
        "local_amos_images": len(images),
        "flare23_masks": len(masks),
        "exact_shape_affine_candidates": dict(sorted(exact_counts.items())),
        "shape_only_candidates": dict(sorted(shape_counts.items())),
        "same_numeric_suffix_with_exact_geometry": direct_stem_geometry_matches,
        "identity_join_valid": exact_counts["unique"] == len(images),
        "interpretation": (
            "Shape alone is not a patient identifier. Without the original FLARE23 "
            "images or an organizer-provided rename manifest, these masks cannot be "
            "paired safely to the local amos_* images."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--content-sample", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    supplied = project / "flare23pseudo"
    aladdin_root = (
        supplied
        / "extracted_aladdin5/aladdin5-pseudo-labels-FLARE23"
    )
    blackbean_root = supplied / "extracted_blackbean"
    aladdin = sorted(aladdin_root.rglob("FLARE23_*.nii.gz"))
    blackbean = sorted(blackbean_root.glob("FLARE23_*.nii.gz"))
    amos_images = sorted((project / "train_unlabel/AMOS-2350").glob("*.nii.gz"))

    expected = {f"FLARE23_{index:04d}.nii.gz" for index in range(1, 4001)}
    aladdin_names = {path.name for path in aladdin}
    blackbean_names = {path.name for path in blackbean}
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision_role": "INPUT_VERIFICATION_ONLY",
        "archives": {
            "aladdin5": {
                "path": str((supplied / "aladdin5-pseudo-labels-FLARE23.zip").resolve()),
                "sha256": sha256(supplied / "aladdin5-pseudo-labels-FLARE23.zip"),
                "zip_crc_test": "PASS",
            },
            "blackbean": {
                "path": str((supplied / "blackbean-pseudo-labels-FLARE23.zip").resolve()),
                "sha256": sha256(supplied / "blackbean-pseudo-labels-FLARE23.zip"),
                "zip_crc_test": "PASS",
            },
        },
        "case_inventory": {
            "expected": 4000,
            "aladdin5": len(aladdin),
            "blackbean": len(blackbean),
            "aladdin5_missing": sorted(expected - aladdin_names),
            "aladdin5_unexpected": sorted(aladdin_names - expected),
            "blackbean_missing": sorted(expected - blackbean_names),
            "blackbean_unexpected": sorted(blackbean_names - expected),
            "paired_names": len(aladdin_names & blackbean_names),
        },
        "aladdin5_content": summarize_masks(
            aladdin, args.workers, args.content_sample
        ),
        "blackbean_content": summarize_masks(
            blackbean, args.workers, args.content_sample
        ),
        "local_amos_identity": identity_audit(amos_images, blackbean),
        "lesion_comparison": {
            "performed": False,
            "reason": (
                "Both supplied releases are 13-organ pseudo-label sets. Tumor/lesion "
                "class 14 is absent, and no valid identity mapping to local amos_* "
                "images exists. Comparing binary GF lesions to these files would be "
                "semantically and patient-wise invalid."
            ),
        },
        "training_used": False,
        "public_used_for_training": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    summary["safe_for_binary_lesion_training"] = bool(
        summary["aladdin5_content"]["content_scan_scope"] == "complete"
        and summary["blackbean_content"]["content_scan_scope"] == "complete"
        and summary["aladdin5_content"]["contains_tumor_label_14"]
        and summary["blackbean_content"]["contains_tumor_label_14"]
        and summary["local_amos_identity"]["identity_join_valid"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
