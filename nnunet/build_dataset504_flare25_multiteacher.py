#!/usr/bin/env python3
"""Build a space-efficient multi-teacher dataset for FLARE25 adaptation.

The 17,575 challenge cases and their preprocessed arrays are reused by symlink.
For Train-Unlabel, a GF component is admitted as a positive target only when
FLARE23 overlaps at least 3% of it. Very large (>=30,000 mm3) GF components are
also retained so that an imperfect confirmer cannot erase obvious tumor burden.
Everything outside per-component context boxes remains unknown in the trainer.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


SCHEMA_VERSION = 1
TARGET_NAME = "Dataset504_FLARE25MultiTeacher"
CACHE_IDENTIFIER = "nnUNetPlans_3d_fullres_S4D2W32"
MIN_OVERLAP_FRACTION = 0.03
LARGE_COMPONENT_MM3 = 30_000.0
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


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


def ensure_symlink(link: Path, source: Path) -> None:
    source = source.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=True) != source:
            raise ValueError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise ValueError(f"refusing to replace existing non-symlink: {link}")
    link.symlink_to(source)


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-6)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-5)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-6)
    )


def select_gf_components(
    gf: np.ndarray,
    flare23: np.ndarray,
    voxel_volume_mm3: float,
    *,
    min_overlap_fraction: float = MIN_OVERLAP_FRACTION,
    large_component_mm3: float = LARGE_COMPONENT_MM3,
) -> tuple[np.ndarray, list[dict]]:
    """Return trusted positive GF components and auditable component records."""

    gf = np.asarray(gf, dtype=bool)
    flare23 = np.asarray(flare23, dtype=bool)
    if gf.shape != flare23.shape or gf.ndim != 3:
        raise ValueError("teacher masks must be aligned 3-D arrays")
    if voxel_volume_mm3 <= 0:
        raise ValueError("voxel volume must be positive")
    labeled, count = ndimage.label(gf, structure=CONNECTIVITY)
    selected = np.zeros(gf.shape, dtype=np.uint8)
    records: list[dict] = []
    for component_id, component_slice in enumerate(
        ndimage.find_objects(labeled, count), start=1
    ):
        if component_slice is None:
            continue
        local = labeled[component_slice] == component_id
        voxels = int(local.sum())
        overlap_voxels = int((local & flare23[component_slice]).sum())
        overlap_fraction = overlap_voxels / max(1, voxels)
        volume_mm3 = voxels * float(voxel_volume_mm3)
        consensus = overlap_fraction >= float(min_overlap_fraction)
        large_rescue = volume_mm3 >= float(large_component_mm3) and not consensus
        keep = consensus or large_rescue
        if keep:
            destination = selected[component_slice]
            destination[local] = 1
        records.append(
            {
                "component_id": component_id,
                "voxels": voxels,
                "volume_mm3": round(volume_mm3, 6),
                "flare23_overlap_voxels": overlap_voxels,
                "flare23_overlap_fraction": round(overlap_fraction, 8),
                "keep": keep,
                "reason": "cross_teacher_consensus" if consensus else (
                    "large_component_recall_rescue" if large_rescue else "unknown"
                ),
            }
        )
    return selected, records


def build_one(task: tuple[str, str, str, str, str]) -> dict:
    identifier, image_text, gf_text, flare23_text, destination_text = task
    image_path, gf_path, flare23_path = map(Path, (image_text, gf_text, flare23_text))
    image = sitk.ReadImage(str(image_path))
    gf_image = sitk.ReadImage(str(gf_path))
    flare23_image = sitk.ReadImage(str(flare23_path))
    if not same_geometry(image, gf_image) or not same_geometry(image, flare23_image):
        raise ValueError(f"teacher geometry mismatch for {identifier}")
    gf = sitk.GetArrayFromImage(gf_image) > 0
    flare23 = sitk.GetArrayFromImage(flare23_image) > 0
    selected, components = select_gf_components(
        gf, flare23, float(np.prod(image.GetSpacing()))
    )
    destination = Path(destination_text)
    selected_voxels = int(selected.sum())
    if selected_voxels:
        if destination.exists():
            existing = sitk.GetArrayFromImage(sitk.ReadImage(str(destination)))
            if not np.array_equal(existing, selected):
                raise ValueError(f"existing target differs: {destination}")
        else:
            output = sitk.GetImageFromArray(selected)
            output.CopyInformation(gf_image)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp.nii.gz"
            )
            sitk.WriteImage(output, str(temporary), True)
            os.replace(temporary, destination)
    return {
        "identifier": identifier,
        "image": str(image_path.resolve()),
        "destination": str(destination.resolve()),
        "components": components,
        "gf_voxels": int(gf.sum()),
        "flare23_voxels": int(flare23.sum()),
        "selected_voxels": selected_voxels,
    }


def base_inventory(raw: Path, cache: Path) -> list[str]:
    images = {
        path.name.removesuffix("_0000.nii.gz")
        for path in (raw / "imagesTr").glob("*_0000.nii.gz")
    }
    labels = {
        path.name.removesuffix(".nii.gz")
        for path in (raw / "labelsTr").glob("*.nii.gz")
    }
    if len(images) != 17_575 or images != labels:
        raise ValueError("Dataset501 raw inventory changed")
    for identifier in images:
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            path = cache / f"{identifier}{suffix}"
            if not path.is_file() or path.stat().st_size < 1:
                raise ValueError(f"missing Dataset501 cache item: {path}")
    return sorted(images)


def pseudo_tasks(root: Path) -> list[tuple[str, str, str, str]]:
    build = json.loads(
        (root / "nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled/"
         "DATASET_BUILD_COMPLETE.json").read_text(encoding="utf-8")
    )
    rows = build.get("pseudo_cases", [])
    if build.get("status") != "COMPLETE" or len(rows) != 2_311:
        raise ValueError("Dataset502 pseudo mapping contract changed")
    flare23_root = root / "external/flare23-aladdin5/train_unlabel_roi_tumor_conf_20260829"
    tasks = []
    for row in rows:
        gf = Path(row["label"]).resolve(strict=True)
        flare23 = (flare23_root / gf.name).resolve(strict=True)
        tasks.append((row["identifier"], row["image"], str(gf), str(flare23)))
    if len({task[0] for task in tasks}) != len(tasks):
        raise ValueError("pseudo identifiers are not unique")
    return tasks


def materialize(root: Path, workers: int) -> dict:
    base_raw = root / "nnUNet_raw/Dataset501_PanCancerCT"
    base_preprocessed = root / "nnUNet_preprocessed/Dataset501_PanCancer"
    target_raw = root / f"nnUNet_raw/{TARGET_NAME}"
    target_preprocessed = root / f"nnUNet_preprocessed/{TARGET_NAME}"
    target_images = target_raw / "imagesTr"
    target_labels = target_raw / "labelsTr"
    target_cache = target_preprocessed / CACHE_IDENTIFIER
    audit_root = target_raw / "teacher_audit"
    for directory in (target_images, target_labels, target_cache, audit_root):
        directory.mkdir(parents=True, exist_ok=True)

    base_ids = base_inventory(base_raw, base_preprocessed / CACHE_IDENTIFIER)
    for identifier in base_ids:
        ensure_symlink(
            target_images / f"{identifier}_0000.nii.gz",
            base_raw / "imagesTr" / f"{identifier}_0000.nii.gz",
        )
        ensure_symlink(
            target_labels / f"{identifier}.nii.gz",
            base_raw / "labelsTr" / f"{identifier}.nii.gz",
        )
        for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
            ensure_symlink(
                target_cache / f"{identifier}{suffix}",
                base_preprocessed / CACHE_IDENTIFIER / f"{identifier}{suffix}",
            )

    tasks = [
        (*task, str(target_labels / f"{task[0]}.nii.gz"))
        for task in pseudo_tasks(root)
    ]
    kept_rows = []
    component_reasons: Counter[str] = Counter()
    cohort_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(build_one, tasks, chunksize=4):
            for component in result["components"]:
                component_reasons[component["reason"]] += 1
            if result["selected_voxels"] == 0:
                continue
            identifier = result["identifier"]
            destination = Path(result["destination"])
            ensure_symlink(
                target_images / f"{identifier}_0000.nii.gz", Path(result["image"])
            )
            cohort = "AMOS" if Path(result["image"]).parts[-2] == "AMOS-2350" else "MSD"
            cohort_counts[cohort] += 1
            kept_rows.append(
                {
                    "identifier": identifier,
                    "cohort": cohort,
                    "image": result["image"],
                    "label": str(destination.resolve()),
                    "label_sha256": sha256(destination),
                    "gf_voxels": result["gf_voxels"],
                    "flare23_voxels": result["flare23_voxels"],
                    "selected_voxels": result["selected_voxels"],
                    "components": result["components"],
                }
            )

    total = len(base_ids) + len(kept_rows)
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": total,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    atomic_json(target_raw / "dataset.json", dataset)
    atomic_json(target_preprocessed / "dataset.json", dataset)
    plans = json.loads((base_preprocessed / "nnUNetPlans.json").read_text())
    plans["dataset_name"] = TARGET_NAME
    atomic_json(target_preprocessed / "nnUNetPlans.json", plans)
    ensure_symlink(
        target_preprocessed / "dataset_fingerprint.json",
        base_preprocessed / "dataset_fingerprint.json",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "policy": {
            "positive_shape": "full_GF_component",
            "cross_teacher_min_overlap_fraction": MIN_OVERLAP_FRACTION,
            "large_component_recall_rescue_mm3": LARGE_COMPONENT_MM3,
            "disputed_and_unmatched_regions": "unknown_not_negative",
        },
        "labeled_cases": len(base_ids),
        "pseudo_cases": len(kept_rows),
        "total_cases": total,
        "pseudo_cohorts": dict(sorted(cohort_counts.items())),
        "component_decisions": dict(sorted(component_reasons.items())),
        "training": kept_rows,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    manifest_path = audit_root / "multiteacher_manifest.json"
    atomic_json(manifest_path, manifest)
    marker = {
        key: manifest[key]
        for key in (
            "schema_version", "created_at", "status", "policy", "labeled_cases",
            "pseudo_cases", "total_cases", "pseudo_cohorts", "component_decisions",
            "public_read", "hidden_read", "submission_created", "uploaded",
        )
    }
    marker.update(
        {
            "target_dataset": TARGET_NAME,
            "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "base_cache_reuse": "symlinked immutable Dataset501 triplets",
            "pseudo_cache_state": "raw targets ready; pseudo cases require preprocessing",
        }
    )
    atomic_json(target_preprocessed / "DATASET_BUILD_COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(materialize(root, args.workers), indent=2))


if __name__ == "__main__":
    main()
