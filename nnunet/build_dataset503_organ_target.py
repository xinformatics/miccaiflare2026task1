#!/usr/bin/env python3
"""Build an isolated focal stomach/bladder pseudo-label continuation dataset.

Dataset501 is immutable. Its raw and preprocessed cases are reused by symlink.
Only conservative AMOS pseudo components that passed the frozen training-only
organ/size/coverage audit are copied into new binary targets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

if __package__:
    from nnunet.build_dataset502_positive_unlabeled import (
        atomic_json,
        base_inventory,
        ensure_symlink,
        sha256,
    )
else:
    from build_dataset502_positive_unlabeled import (  # type: ignore[no-redef]
        atomic_json,
        base_inventory,
        ensure_symlink,
        sha256,
    )


TARGET_NAME = "Dataset503_PanCancerOrganTarget"
CACHE_IDENTIFIER = "nnUNetPlans_3d_fullres_S4D2W32"
EXPECTED_BASE_CASES = 17_575


def selected_components(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected: defaultdict[str, list[int]] = defaultdict(list)
    for row in rows:
        if str(row["target_eligible"]).strip().lower() not in {"true", "1"}:
            continue
        if row["dominant_organ"] not in {"stomach", "urinary_bladder"}:
            raise ValueError("eligible component has a non-target organ")
        selected[row["case"]].append(int(row["component"]))
    result = {case: tuple(sorted(set(values))) for case, values in sorted(selected.items())}
    if not result or any(not values for values in result.values()):
        raise ValueError("organ-target audit selected no usable components")
    return result


def manifest_entries(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("training")
    if not isinstance(rows, list):
        raise ValueError("conservative pseudo manifest has no training list")
    return {
        Path(row["image"]).name.removesuffix("_0000.nii.gz"): row
        for row in rows
        if "AMOS-2350" in Path(row["image"]).parts
    }


def filtered_label(
    pseudo_path: Path,
    component_ids: tuple[int, ...],
    destination: Path,
) -> dict:
    image = nib.load(str(pseudo_path))
    mask = np.asanyarray(image.dataobj) > 0
    coordinates = np.where(mask)
    if not coordinates[0].size:
        raise ValueError(f"conservative pseudo label is empty: {pseudo_path}")
    box = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
    cropped = mask[box]
    components, count = ndimage.label(
        cropped, structure=ndimage.generate_binary_structure(3, 1)
    )
    if max(component_ids) > count:
        raise ValueError(f"component selection exceeds inventory for {pseudo_path}")
    filtered_crop = np.isin(components, component_ids)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    filtered[box] = filtered_crop
    if not filtered.any():
        raise ValueError(f"filtered label is empty: {pseudo_path}")

    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(filtered, image.affine, header), str(temporary))
    if destination.exists():
        if sha256(destination) != sha256(temporary):
            temporary.unlink()
            raise ValueError(f"existing filtered target differs: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    return {
        "voxels": int(filtered.sum()),
        "volume_mm3": float(filtered.sum() * np.prod(spacing)),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument(
        "--audit-components",
        type=Path,
        default=root / "nnunet/analysis/amos22_conservative_pseudo_anatomy_20260827/components.csv",
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=root / "nnunet/analysis/amos22_conservative_pseudo_anatomy_20260827/summary.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            root
            / "external/GF-Screen/public_sweeps/gf_train_unlabel_cbg_direct_20260825/"
            "conservative/manifest.json"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    audit_components = args.audit_components.resolve()
    audit_summary = args.audit_summary.resolve()
    manifest_path = args.manifest.resolve()
    selection = selected_components(audit_components)
    summary = json.loads(audit_summary.read_text(encoding="utf-8"))
    policy = summary.get("target_component_policy", {})
    if (
        summary.get("status") != "COMPLETE"
        or int(policy.get("eligible_components", -1)) != sum(map(len, selection.values()))
        or int(policy.get("eligible_cases", -1)) != len(selection)
        or summary.get("public_read") is not False
        or summary.get("hidden_read") is not False
    ):
        raise ValueError("organ-target audit contract is invalid")

    entries = manifest_entries(manifest_path)
    missing = sorted(set(selection) - set(entries))
    if missing:
        raise ValueError(f"selected cases absent from conservative manifest: {missing}")

    base_raw = root / "nnUNet_raw/Dataset501_PanCancerCT"
    base_preprocessed = root / "nnUNet_preprocessed/Dataset501_PanCancer"
    target_raw = root / f"nnUNet_raw/{TARGET_NAME}"
    target_preprocessed = root / f"nnUNet_preprocessed/{TARGET_NAME}"
    base_ids = base_inventory(base_raw, base_preprocessed / CACHE_IDENTIFIER)
    if len(base_ids) != EXPECTED_BASE_CASES:
        raise ValueError("Dataset501 base inventory changed")

    selected_rows = []
    for index, case in enumerate(sorted(selection), start=1):
        entry = entries[case]
        selected_rows.append(
            {
                "identifier": f"PseudoOrgan-AMOS22_{index:04d}",
                "case": case,
                "component_ids": list(selection[case]),
                "image": str(Path(entry["image"]).resolve()),
                "conservative_label": str(Path(entry["label"]).resolve()),
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "VALIDATED_DRY_RUN",
        "target_dataset": TARGET_NAME,
        "labeled_cases": len(base_ids),
        "organ_target_cases": len(selected_rows),
        "organ_target_components": sum(len(row["component_ids"]) for row in selected_rows),
        "total_cases": len(base_ids) + len(selected_rows),
        "selection": selected_rows,
        "selection_policy": policy,
        "inputs": {
            "audit_components": {"path": str(audit_components), "sha256": sha256(audit_components)},
            "audit_summary": {"path": str(audit_summary), "sha256": sha256(audit_summary)},
            "conservative_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        },
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    if args.materialize:
        target_images = target_raw / "imagesTr"
        target_labels = target_raw / "labelsTr"
        target_cache = target_preprocessed / CACHE_IDENTIFIER
        for directory in (target_images, target_labels, target_cache):
            directory.mkdir(parents=True, exist_ok=True)
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
        for row in selected_rows:
            identifier = row["identifier"]
            ensure_symlink(target_images / f"{identifier}_0000.nii.gz", Path(row["image"]))
            row["filtered_label"] = str(target_labels / f"{identifier}.nii.gz")
            row["filtered_label_stats"] = filtered_label(
                Path(row["conservative_label"]),
                tuple(row["component_ids"]),
                Path(row["filtered_label"]),
            )

        dataset = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "tumor": 1},
            "numTraining": len(base_ids) + len(selected_rows),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        atomic_json(target_raw / "dataset.json", dataset)
        atomic_json(target_preprocessed / "dataset.json", dataset)
        plans = json.loads((base_preprocessed / "nnUNetPlans.json").read_text(encoding="utf-8"))
        if plans.get("dataset_name") != "Dataset501_PanCancer":
            raise ValueError("Dataset501 numerical plan identity changed")
        plans["dataset_name"] = TARGET_NAME
        atomic_json(target_preprocessed / "nnUNetPlans.json", plans)
        ensure_symlink(
            target_preprocessed / "dataset_fingerprint.json",
            base_preprocessed / "dataset_fingerprint.json",
        )
        payload["status"] = "COMPLETE"
        payload["base_cache_reuse"] = "symlinked immutable Dataset501 triplets"
        payload["organ_target_cache_state"] = "raw filtered targets ready; target cases require preprocessing"
        terminal = target_preprocessed / "DATASET_BUILD_COMPLETE.json"
        atomic_json(terminal, payload)

    if args.output:
        atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
