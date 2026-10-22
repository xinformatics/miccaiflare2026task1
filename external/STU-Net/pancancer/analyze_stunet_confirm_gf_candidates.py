#!/usr/bin/env python3
"""DSC-first screen for STU-Net-confirmed high-value GF components.

This diagnostic never materializes a prediction and never reads hidden data.
It asks one narrow question on the fixed public set: can an independent
STU-Net mask safely confirm disconnected, whole GF components at or above
3,000 mm3?  Only candidates clearing the immutable +0.015 public-DSC gate are
worth the substantially more expensive exact NSD/materialization pass.

The implementation labels only the tight GF foreground extent and releases
full-volume arrays between statistics.  That keeps the largest 512x512x554
public cases below the memory footprint of the older dense component audits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


CONNECTIVITY = 6
MINIMUM_COMPONENT_MM3 = 3000.0
SUPPORT_FRACTIONS = (0.01, 0.10)
DSC_GATE = 0.6154316379500749
EXPECTED_ANCHOR_DSC = 0.6004316379500749


@dataclass(frozen=True)
class CaseCounts:
    anchor_dsc: float
    candidate_dsc: dict[str, float]
    selected_components: dict[str, int]
    true_selected_components: dict[str, int]
    selected_voxels: dict[str, int]
    selected_true_positive_voxels: dict[str, int]


def sha256_file(path: Path) -> str:
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


def foreground_extent(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return None
    return tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates
    )


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    observed = np.unique(values)
    if not np.isin(observed, (0, 1)).all():
        raise ValueError(f"nonbinary mask {path}: {observed[:20].tolist()}")
    return image, values.astype(bool, copy=False)


def validate_geometry(
    name: str, reference: nib.Nifti1Image, candidate: nib.Nifti1Image
) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, atol=1e-4
    ):
        raise ValueError(f"geometry mismatch for {name}")


def dice_from_counts(tp: int, prediction: int, target: int) -> float:
    denominator = prediction + target
    return float(2 * tp / denominator) if denominator else 1.0


def counts_from_component_statistics(
    *,
    sizes: np.ndarray,
    anchor_overlap: np.ndarray,
    support_overlap: np.ndarray,
    target_overlap: np.ndarray,
    voxel_mm3: float,
    anchor_voxels: int,
    target_voxels: int,
    anchor_tp: int,
) -> CaseCounts:
    """Apply the fixed confirmation grid to already aggregated components."""

    component_count = len(sizes) - 1
    for role, values in (
        ("anchor_overlap", anchor_overlap),
        ("support_overlap", support_overlap),
        ("target_overlap", target_overlap),
    ):
        if len(values) != len(sizes):
            raise ValueError(f"{role}/size length mismatch")
    anchor_dsc = dice_from_counts(anchor_tp, anchor_voxels, target_voxels)
    eligible = (anchor_overlap == 0) & (
        sizes.astype(np.float64) * voxel_mm3 >= MINIMUM_COMPONENT_MM3
    )
    eligible[0] = False
    support_fraction = np.divide(
        support_overlap,
        sizes,
        out=np.zeros(component_count + 1, dtype=np.float64),
        where=sizes > 0,
    )
    candidate_dsc: dict[str, float] = {}
    selected_components: dict[str, int] = {}
    true_selected_components: dict[str, int] = {}
    selected_voxels: dict[str, int] = {}
    selected_tp: dict[str, int] = {}
    for minimum_support in SUPPORT_FRACTIONS:
        tag = f"stu_s{int(round(minimum_support * 100)):02d}"
        keep = eligible & (support_fraction >= minimum_support)
        ids = np.flatnonzero(keep)
        addition_voxels = int(sizes[ids].sum())
        addition_tp = int(target_overlap[ids].sum())
        candidate_dsc[tag] = dice_from_counts(
            anchor_tp + addition_tp,
            anchor_voxels + addition_voxels,
            target_voxels,
        )
        selected_components[tag] = int(ids.size)
        true_selected_components[tag] = int(np.count_nonzero(target_overlap[ids]))
        selected_voxels[tag] = addition_voxels
        selected_tp[tag] = addition_tp
    return CaseCounts(
        anchor_dsc=anchor_dsc,
        candidate_dsc=candidate_dsc,
        selected_components=selected_components,
        true_selected_components=true_selected_components,
        selected_voxels=selected_voxels,
        selected_true_positive_voxels=selected_tp,
    )


def component_confirmation_counts(
    *,
    candidate: np.ndarray,
    anchor: np.ndarray,
    support: np.ndarray,
    target: np.ndarray,
    voxel_mm3: float,
) -> CaseCounts:
    """Return per-configuration DSC without constructing full prediction masks."""

    if not (candidate.shape == anchor.shape == support.shape == target.shape):
        raise ValueError("mask shape mismatch")
    extent = foreground_extent(candidate)
    anchor_voxels = int(anchor.sum())
    target_voxels = int(target.sum())
    anchor_tp = int(np.count_nonzero(anchor & target))
    anchor_dsc = dice_from_counts(anchor_tp, anchor_voxels, target_voxels)
    tags = tuple(f"stu_s{int(round(value * 100)):02d}" for value in SUPPORT_FRACTIONS)
    if extent is None:
        empty_int = {tag: 0 for tag in tags}
        return CaseCounts(
            anchor_dsc=anchor_dsc,
            candidate_dsc={tag: anchor_dsc for tag in tags},
            selected_components=empty_int.copy(),
            true_selected_components=empty_int.copy(),
            selected_voxels=empty_int.copy(),
            selected_true_positive_voxels=empty_int.copy(),
        )

    candidate_crop = np.ascontiguousarray(candidate[extent], dtype=np.uint8)
    labels, component_count = cc3d.connected_components(
        candidate_crop,
        connectivity=CONNECTIVITY,
        return_N=True,
        out_dtype=np.uint16,
        binary_image=True,
    )
    if component_count >= np.iinfo(np.uint16).max:
        raise ValueError("candidate component count exceeds uint16 contract")
    sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    anchor_overlap = np.bincount(
        labels[anchor[extent]], minlength=component_count + 1
    )
    support_overlap = np.bincount(
        labels[support[extent]], minlength=component_count + 1
    )
    target_overlap = np.bincount(
        labels[target[extent]], minlength=component_count + 1
    )
    return counts_from_component_statistics(
        sizes=sizes,
        anchor_overlap=anchor_overlap,
        support_overlap=support_overlap,
        target_overlap=target_overlap,
        voxel_mm3=voxel_mm3,
        anchor_voxels=anchor_voxels,
        target_voxels=target_voxels,
        anchor_tp=anchor_tp,
    )


def component_confirmation_counts_from_paths(
    *,
    candidate_path: Path,
    anchor_path: Path,
    support_path: Path,
    target_path: Path,
) -> CaseCounts:
    """Path-based variant that bounds live full-volume masks to two at a time."""

    candidate_image, candidate = load_binary(candidate_path)
    extent = foreground_extent(candidate)
    voxel_mm3 = float(abs(np.linalg.det(candidate_image.affine[:3, :3])))
    if extent is None:
        sizes = np.zeros(1, dtype=np.int64)
        labels = np.zeros((0, 0, 0), dtype=np.uint16)
        component_count = 0
    else:
        candidate_crop = np.ascontiguousarray(candidate[extent], dtype=np.uint8)
        labels, component_count = cc3d.connected_components(
            candidate_crop,
            connectivity=CONNECTIVITY,
            return_N=True,
            out_dtype=np.uint16,
            binary_image=True,
        )
        sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    del candidate

    anchor_image, anchor = load_binary(anchor_path)
    validate_geometry(f"anchor/{anchor_path.name}", candidate_image, anchor_image)
    anchor_voxels = int(anchor.sum())
    anchor_overlap = (
        np.bincount(labels[anchor[extent]], minlength=component_count + 1)
        if extent is not None
        else np.zeros(1, dtype=np.int64)
    )

    target_image, target = load_binary(target_path)
    validate_geometry(f"target/{target_path.name}", candidate_image, target_image)
    target_voxels = int(target.sum())
    anchor_tp = int(np.count_nonzero(anchor & target))
    target_overlap = (
        np.bincount(labels[target[extent]], minlength=component_count + 1)
        if extent is not None
        else np.zeros(1, dtype=np.int64)
    )
    del anchor, target

    support_image, support = load_binary(support_path)
    validate_geometry(f"support/{support_path.name}", candidate_image, support_image)
    support_overlap = (
        np.bincount(labels[support[extent]], minlength=component_count + 1)
        if extent is not None
        else np.zeros(1, dtype=np.int64)
    )
    del support, labels
    return counts_from_component_statistics(
        sizes=sizes,
        anchor_overlap=anchor_overlap,
        support_overlap=support_overlap,
        target_overlap=target_overlap,
        voxel_mm3=voxel_mm3,
        anchor_voxels=anchor_voxels,
        target_voxels=target_voxels,
        anchor_tp=anchor_tp,
    )


def inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.nii.gz")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--support", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = sorted(inventory(args.labels))
    if len(names) != 50:
        raise ValueError(f"fixed-public inventory must contain 50 cases, got {len(names)}")
    expected = set(names)
    for role, directory in (
        ("anchor", args.anchor),
        ("candidate", args.candidate),
        ("support", args.support),
    ):
        observed = inventory(directory)
        if observed != expected:
            raise ValueError(
                f"{role} inventory mismatch: missing={sorted(expected-observed)} "
                f"extra={sorted(observed-expected)}"
            )

    records = []
    for index, name in enumerate(names, 1):
        counts = component_confirmation_counts_from_paths(
            candidate_path=args.candidate / name,
            anchor_path=args.anchor / name,
            support_path=args.support / name,
            target_path=args.labels / name,
        )
        records.append(
            {
                "case": name.removesuffix(".nii.gz"),
                "anchor_dsc": counts.anchor_dsc,
                "candidate_dsc": counts.candidate_dsc,
                "selected_components": counts.selected_components,
                "true_selected_components": counts.true_selected_components,
                "selected_voxels": counts.selected_voxels,
                "selected_true_positive_voxels": counts.selected_true_positive_voxels,
            }
        )
        print(f"[{index:02d}/50] {name}", flush=True)

    anchor_dsc = float(np.mean([row["anchor_dsc"] for row in records]))
    if not np.isclose(anchor_dsc, EXPECTED_ANCHOR_DSC, atol=1e-12, rtol=0):
        raise ValueError(
            f"anchor DSC identity mismatch: {anchor_dsc} != {EXPECTED_ANCHOR_DSC}"
        )
    configurations = []
    for support_fraction in SUPPORT_FRACTIONS:
        tag = f"stu_s{int(round(support_fraction * 100)):02d}"
        dsc = float(np.mean([row["candidate_dsc"][tag] for row in records]))
        configurations.append(
            {
                "configuration": tag,
                "minimum_candidate_mm3": MINIMUM_COMPONENT_MM3,
                "minimum_stunet_support_fraction": support_fraction,
                "Lesion_DSC": dsc,
                "DSC_gain": dsc - anchor_dsc,
                "selected_components": int(
                    sum(row["selected_components"][tag] for row in records)
                ),
                "true_selected_components": int(
                    sum(row["true_selected_components"][tag] for row in records)
                ),
                "selected_voxels": int(
                    sum(row["selected_voxels"][tag] for row in records)
                ),
                "selected_true_positive_voxels": int(
                    sum(row["selected_true_positive_voxels"][tag] for row in records)
                ),
                "passes_dsc_gate": dsc >= DSC_GATE,
            }
        )
    configurations.sort(key=lambda row: row["Lesion_DSC"], reverse=True)
    selected = next(
        (row for row in configurations if row["passes_dsc_gate"]), None
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "QUALIFY_EXACT_NSD_MATERIALIZATION" if selected else "KEEP_TOUCHCOMP",
        "selection_policy": (
            "maximum fixed-public DSC among predeclared 1%/10% STU-Net support "
            "fractions; exact NSD is deferred until the immutable DSC gate passes"
        ),
        "connectivity": CONNECTIVITY,
        "anchor_DSC": anchor_dsc,
        "fixed_public_DSC_gate": DSC_GATE,
        "selected": selected,
        "configurations": configurations,
        "cases": records,
        "assets": {
            role: {"path": str(path.resolve())}
            for role, path in (
                ("labels", args.labels),
                ("anchor", args.anchor),
                ("candidate", args.candidate),
                ("support", args.support),
            )
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("decision", "anchor_DSC", "selected", "configurations")}, indent=2))


if __name__ == "__main__":
    main()
