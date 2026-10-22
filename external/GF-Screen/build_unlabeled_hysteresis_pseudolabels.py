#!/usr/bin/env python3
"""Build high-confidence lesion pseudo labels from nested teacher masks.

The lower-threshold mask supplies lesion extent. A connected support component
is retained only when it contains a sufficiently large higher-threshold seed.
Teacher-empty images are recorded but excluded from the positive-only training
manifest, so they can never become hard-negative ground truth.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np

from inference_naming import prediction_name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_images(root: Path, excluded=()):
    excluded = {Path(path).resolve() for path in excluded}
    images = sorted(
        path for path in root.rglob("*.nii.gz")
        if path.is_file() and path.resolve() not in excluded
    )
    by_prediction = {}
    for path in images:
        key = prediction_name(path.name)
        if key in by_prediction:
            raise ValueError(f"duplicate prediction basename: {key}")
        by_prediction[key] = path
    return by_prediction


def build_case(support_path, seed_path, output_path, min_seed_mm3, min_component_mm3):
    support_image = nib.load(support_path)
    seed_image = nib.load(seed_path)
    if support_image.shape != seed_image.shape or not np.allclose(
        support_image.affine, seed_image.affine, atol=1e-4
    ):
        raise ValueError(f"support/seed geometry mismatch: {support_path.name}")
    support = np.asarray(support_image.dataobj) > 0
    seed = np.asarray(seed_image.dataobj) > 0
    if np.any(seed & ~support):
        raise ValueError(f"seed is not nested inside support: {support_path.name}")
    voxel_mm3 = float(abs(np.linalg.det(support_image.affine[:3, :3])))
    locations = np.nonzero(support)
    if not locations[0].size:
        return {
            "components": 0,
            "volume_mm3": 0.0,
            "voxel_mm3": voxel_mm3,
        }
    bounds = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in locations
    )
    support_crop = support[bounds]
    seed_crop = seed[bounds]
    components, count = cc3d.connected_components(
        support_crop, connectivity=26, return_N=True
    )
    identifiers = np.arange(1, count + 1)
    component_mm3 = (
        np.bincount(components.reshape(-1), minlength=count + 1)[1:].astype(float)
        * voxel_mm3
    )
    seed_mm3 = (
        np.bincount(components[seed_crop], minlength=count + 1)[1:].astype(float)
        * voxel_mm3
    )
    retained_ids = identifiers[
        (seed_mm3 >= min_seed_mm3) & (component_mm3 >= min_component_mm3)
    ]
    retained_components = int(retained_ids.size)
    retained_mm3 = float(component_mm3[retained_ids - 1].sum())
    if retained_components:
        keep = np.zeros(count + 1, dtype=bool)
        keep[retained_ids] = True
        retained = np.zeros(support.shape, dtype=np.uint8)
        retained[bounds] = keep[components].astype(np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp.nii.gz")
        nib.save(nib.Nifti1Image(retained, support_image.affine, support_image.header), temporary)
        temporary.replace(output_path)
    return {
        "components": int(retained_components),
        "volume_mm3": retained_mm3,
        "voxel_mm3": voxel_mm3,
    }


def build_case_record(task):
    (
        key, image_path, support_path, seed_path, output_path,
        min_seed_mm3, min_component_mm3, recipe,
    ) = task
    stats = build_case(
        support_path, seed_path, output_path,
        min_seed_mm3, min_component_mm3,
    )
    case = {
        "name": key.removesuffix(".nii.gz"),
        "image": str(image_path.resolve()),
        "label": str(output_path.resolve()) if stats["components"] else None,
        **stats,
    }
    training = None
    if stats["components"]:
        case["label_sha256"] = file_sha256(output_path)
        case["label_bytes"] = output_path.stat().st_size
        training = {
            "image": case["image"],
            "label": case["label"],
            "label_sha256": case["label_sha256"],
            "label_bytes": case["label_bytes"],
            "name": f"Pseudo-{recipe}_{case['name']}",
            "pseudo_label": True,
        }
    return case, training


def build_records(tasks, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [build_case_record(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(build_case_record, tasks, chunksize=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument(
        "--exclude-audit",
        type=Path,
        help="Train-Unlabel audit whose exact-overlap files must be excluded",
    )
    parser.add_argument("--min-seed-mm3", type=float, required=True)
    parser.add_argument("--min-component-mm3", type=float, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--workers", type=int, default=8,
        help="parallel NIfTI decode/component workers (default: 8)",
    )
    args = parser.parse_args()
    if args.min_seed_mm3 <= 0 or args.min_component_mm3 <= 0:
        parser.error("physical-volume thresholds must be positive")
    if args.workers < 1:
        parser.error("workers must be positive")

    excluded = []
    if args.exclude_audit:
        exclusion_payload = json.loads(args.exclude_audit.read_text())
        excluded = exclusion_payload.get("excluded_unlabeled_files", [])
    excluded = sorted(str(Path(path).resolve()) for path in excluded)
    images = discover_images(args.images, excluded=excluded)
    support_paths = {path.name: path for path in args.support.glob("*.nii.gz")}
    seed_paths = {path.name: path for path in args.seed.glob("*.nii.gz")}
    expected = set(images)
    if not expected <= set(support_paths) or not expected <= set(seed_paths):
        raise ValueError(
            "teacher prediction inventory mismatch: "
            f"images={len(expected)} support={len(support_paths)} seed={len(seed_paths)}"
        )

    tasks = [
        (
            key, images[key], support_paths[key], seed_paths[key],
            args.output_labels / key, args.min_seed_mm3,
            args.min_component_mm3, args.recipe,
        )
        for key in sorted(expected)
    ]
    records = build_records(tasks, args.workers)
    cases = [case for case, _ in records]
    training = [item for _, item in records if item is not None]

    manifest = {
        "training": training,
        "validation": [],
        "description": (
            "Positive-only teacher hysteresis pseudo labels. Missing entries "
            "are unknown, never hard-negative cases."
        ),
    }
    audit = {
        "schema": "gf-unlabeled-hysteresis-v2",
        "recipe": args.recipe,
        "images": len(images),
        "excluded_exact_overlaps": len(excluded),
        "excluded_unlabeled_files": excluded,
        "positive_cases": len(training),
        "empty_or_rejected_cases": len(images) - len(training),
        "components": sum(case["components"] for case in cases),
        "volume_mm3": sum(case["volume_mm3"] for case in cases),
        "min_seed_mm3": args.min_seed_mm3,
        "min_component_mm3": args.min_component_mm3,
        "workers": args.workers,
        "cases": cases,
    }
    for path, payload in (
        (args.output_manifest, manifest), (args.output_audit, audit)
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    print(json.dumps({key: audit[key] for key in (
        "recipe", "images", "positive_cases", "empty_or_rejected_cases",
        "components", "volume_mm3",
    )}, indent=2))


if __name__ == "__main__":
    main()
