#!/usr/bin/env python3
"""Audit accepted AMOS pseudo-lesions against official AMOS22 organ masks.

The official organ masks are used only to describe and potentially stratify
accepted pseudo-positive cases. They are not lesion targets, and rejected
Train-Unlabel cases remain unknown rather than negative.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from nibabel.processing import resample_from_to


AMOS_LABELS = {
    0: "background",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "duodenum",
    14: "urinary_bladder",
    15: "prostate_or_uterus",
}
TARGET_ORGANS = {7, 14}
TARGET_MIN_MM3 = 1_000.0
TARGET_MAX_MM3 = 50_000.0
TARGET_MIN_COMPONENT_FRACTION = 0.60
TARGET_MAX_ORGAN_COVERAGE = 0.20


def is_target_eligible(
    dominant_label: int,
    volume_mm3: float,
    component_fraction: float,
    organ_coverage: float,
) -> bool:
    """Apply the frozen anatomy-target policy to one connected component."""

    return bool(
        dominant_label in TARGET_ORGANS
        and TARGET_MIN_MM3 <= volume_mm3 <= TARGET_MAX_MM3
        and component_fraction >= TARGET_MIN_COMPONENT_FRACTION
        and organ_coverage <= TARGET_MAX_ORGAN_COVERAGE
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "external/GF-Screen/public_sweeps/"
            "gf_train_unlabel_cbg_direct_20260825/balanced/manifest.json"
        ),
    )
    parser.add_argument(
        "--organ-labels",
        type=Path,
        default=Path("external/AMOS22_CT_LABELS_20260827"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nnunet/analysis/amos22_positive_pseudo_anatomy_20260827"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    organ_root = args.organ_labels.resolve()
    output_dir = args.output.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["training"] if isinstance(manifest, dict) else manifest
    accepted = {
        Path(entry["image"]).name.replace("_0000.nii.gz", ".nii.gz"): entry
        for entry in entries
        if "AMOS-2350" in Path(entry["image"]).parts
    }
    organ_paths = sorted(organ_root.glob("*/labels*/*.nii.gz"))
    if len(organ_paths) != 300:
        raise ValueError(f"expected 300 AMOS22 CT organ labels, found {len(organ_paths)}")

    structure = ndimage.generate_binary_structure(3, 1)
    component_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    matched = 0
    reoriented_organ_labels = 0
    resampled_organ_labels = 0
    for organ_path in organ_paths:
        entry = accepted.get(organ_path.name)
        if entry is None:
            continue
        matched += 1
        pseudo_path = Path(entry["label"]).resolve()
        organ_img = nib.load(str(organ_path))
        pseudo_img = nib.load(str(pseudo_path))
        pseudo = np.asanyarray(pseudo_img.dataobj) > 0
        if organ_img.shape != pseudo.shape or not np.allclose(organ_img.affine, pseudo_img.affine):
            canonical = nib.as_closest_canonical(organ_img)
            if canonical.shape == pseudo.shape and np.allclose(canonical.affine, pseudo_img.affine):
                organ_img = canonical
                reoriented_organ_labels += 1
            else:
                organ_img = resample_from_to(organ_img, (pseudo_img.shape, pseudo_img.affine), order=0)
                resampled_organ_labels += 1
        organ = np.rint(np.asanyarray(organ_img.dataobj)).astype(np.int16, copy=False)
        if organ.shape != pseudo.shape:
            raise ValueError(f"unresolved geometry mismatch for {organ_path.name}")
        organ_voxel_counts = np.bincount(organ.ravel(), minlength=16)
        spacing = np.asarray(pseudo_img.header.get_zooms()[:3], dtype=float)
        voxel_mm3 = float(np.prod(spacing))
        foreground_coordinates = np.where(pseudo)
        foreground_box = tuple(
            slice(int(axis.min()), int(axis.max()) + 1) for axis in foreground_coordinates
        )
        pseudo_crop = pseudo[foreground_box]
        organ_crop = organ[foreground_box]
        components, count = ndimage.label(pseudo_crop, structure=structure)
        if count <= 0:
            raise ValueError(f"accepted pseudo-positive case is empty: {pseudo_path}")

        case_counts = np.bincount(organ[pseudo].ravel(), minlength=16)
        nonzero_case_counts = case_counts[1:]
        dominant_case_label = int(np.argmax(nonzero_case_counts) + 1) if nonzero_case_counts.max() else 0
        case_rows.append(
            {
                "case": organ_path.name.removesuffix(".nii.gz"),
                "pseudo_components": count,
                "pseudo_voxels": int(pseudo.sum()),
                "pseudo_mm3": float(pseudo.sum() * voxel_mm3),
                "dominant_organ": AMOS_LABELS[dominant_case_label],
                "dominant_organ_fraction": (
                    float(case_counts[dominant_case_label] / pseudo.sum()) if dominant_case_label else 0.0
                ),
                "stomach_fraction": float(case_counts[7] / pseudo.sum()),
                "urinary_bladder_fraction": float(case_counts[14] / pseudo.sum()),
                "outside_labeled_organs_fraction": float(case_counts[0] / pseudo.sum()),
            }
        )

        objects = ndimage.find_objects(components)
        for component_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            local = components[slices] == component_id
            voxels = int(local.sum())
            local_organs = organ_crop[slices][local]
            counts = np.bincount(local_organs.ravel(), minlength=16)
            nonzero_counts = counts[1:]
            dominant_label = int(np.argmax(nonzero_counts) + 1) if nonzero_counts.max() else 0
            component_fraction = float(counts[dominant_label] / voxels) if dominant_label else 0.0
            organ_coverage = (
                float(counts[dominant_label] / organ_voxel_counts[dominant_label])
                if dominant_label and organ_voxel_counts[dominant_label] > 0
                else 0.0
            )
            volume_mm3 = voxels * voxel_mm3
            target_eligible = is_target_eligible(
                dominant_label, volume_mm3, component_fraction, organ_coverage
            )
            component_rows.append(
                {
                    "case": organ_path.name.removesuffix(".nii.gz"),
                    "component": component_id,
                    "voxels": voxels,
                    "volume_mm3": volume_mm3,
                    "dominant_organ": AMOS_LABELS[dominant_label],
                    "dominant_organ_fraction": component_fraction,
                    "fraction_of_dominant_organ_covered": organ_coverage,
                    "stomach_fraction": float(counts[7] / voxels),
                    "urinary_bladder_fraction": float(counts[14] / voxels),
                    "outside_labeled_organs_fraction": float(counts[0] / voxels),
                    "target_eligible": target_eligible,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    components_path = output_dir / "components.csv"
    cases_path = output_dir / "cases.csv"
    for path, rows in ((components_path, component_rows), (cases_path, case_rows)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    by_organ: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in component_rows:
        by_organ[str(row["dominant_organ"])].append(row)
    organ_summary = {}
    for organ, rows in sorted(by_organ.items()):
        organ_summary[organ] = {
            "dominant_components": len(rows),
            "cases": len({str(row["case"]) for row in rows}),
            "total_component_mm3": float(sum(float(row["volume_mm3"]) for row in rows)),
            "median_component_mm3": float(np.median([float(row["volume_mm3"]) for row in rows])),
            "median_dominant_fraction": float(
                np.median([float(row["dominant_organ_fraction"]) for row in rows])
            ),
        }

    label_manifest = "\n".join(f"{path.name} {sha256(path)}" for path in organ_paths)
    eligible_rows = [row for row in component_rows if bool(row["target_eligible"])]
    summary = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "purpose": "training-only anatomy audit of accepted AMOS Train-Unlabel pseudo-lesions",
        "official_amos22_ct_cases": len(organ_paths),
        "accepted_positive_pseudo_cases_matched": matched,
        "official_cases_without_accepted_positive_pseudo": len(organ_paths) - matched,
        "accepted_component_count": len(component_rows),
        "organ_labels_reoriented_to_pseudo_grid": reoriented_organ_labels,
        "organ_labels_resampled_to_pseudo_grid": resampled_organ_labels,
        "organs": organ_summary,
        "target_organs": {
            organ: organ_summary.get(
                organ,
                {
                    "dominant_components": 0,
                    "cases": 0,
                    "total_component_mm3": 0.0,
                    "median_component_mm3": None,
                    "median_dominant_fraction": None,
                },
            )
            for organ in ("stomach", "urinary_bladder")
        },
        "target_component_policy": {
            "organs": [AMOS_LABELS[label] for label in sorted(TARGET_ORGANS)],
            "minimum_mm3": TARGET_MIN_MM3,
            "maximum_mm3": TARGET_MAX_MM3,
            "minimum_component_fraction_inside_organ": TARGET_MIN_COMPONENT_FRACTION,
            "maximum_fraction_of_organ_covered": TARGET_MAX_ORGAN_COVERAGE,
            "eligible_components": len(eligible_rows),
            "eligible_cases": len({str(row["case"]) for row in eligible_rows}),
            "eligible_component_ids": [
                f"{row['case']}:{row['component']}" for row in eligible_rows
            ],
        },
        "inputs": {
            "pseudo_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "organ_labels": {
                "path": str(organ_root),
                "file_count": len(organ_paths),
                "manifest_sha256": hashlib.sha256(label_manifest.encode()).hexdigest(),
                "source": "MedOtter/amos22-ct-dataset (CC-BY-4.0 mirror of AMOS22)",
            },
        },
        "tables": {
            "cases": {"path": str(cases_path), "sha256": sha256(cases_path)},
            "components": {"path": str(components_path), "sha256": sha256(components_path)},
        },
        "interpretation_guard": (
            "organ masks describe accepted pseudo-labels only; they are not lesion supervision, "
            "and the 55 rejected/unselected AMOS22 cases remain unknown rather than negative"
        ),
        "public_read": False,
        "hidden_read": False,
        "training_mutated": False,
        "submission_created": False,
        "uploaded": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
