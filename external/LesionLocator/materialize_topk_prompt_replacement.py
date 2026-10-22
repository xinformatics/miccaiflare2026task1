#!/usr/bin/env python3
"""Replace only prompted anchor components with gated LesionLocator outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


NIFTI_SUFFIX = ".nii.gz"


def case_name(reference: Path) -> str:
    stem = reference.name[: -len(NIFTI_SUFFIX)]
    return stem[:-5] if stem.endswith("_0000") else stem


def same_geometry(left: nib.Nifti1Image, right: nib.Nifti1Image) -> bool:
    return left.shape == right.shape and np.allclose(left.affine, right.affine, atol=1e-4, rtol=0)


def instance_union(reference: Path, instances: Path) -> np.ndarray:
    ref = nib.load(reference)
    result = np.zeros(ref.shape, dtype=bool)
    prefix = reference.name[: -len(NIFTI_SUFFIX)] + "_lesion_"
    candidates = sorted(
        instances.glob(prefix + "*" + NIFTI_SUFFIX),
        key=lambda path: int(re.search(r"_lesion_(\d+)\.nii\.gz$", path.name).group(1)),
    )
    for path in candidates:
        image = nib.load(path)
        if not same_geometry(ref, image):
            raise ValueError(f"instance geometry mismatch: {path}")
        result |= np.asarray(image.dataobj) > 0
    return result


def gated_replacement(anchor: np.ndarray, prompt: np.ndarray, refined: np.ndarray, min_seed_coverage: float, min_refined_overlap: float, min_volume_ratio: float, max_volume_ratio: float) -> tuple[np.ndarray, dict[str, object]]:
    if anchor.shape != prompt.shape or anchor.shape != refined.shape:
        raise ValueError("replacement array shape mismatch")
    if np.any(prompt & ~anchor):
        raise ValueError("prompt contains voxels outside frozen anchor")
    seed_voxels = int(prompt.sum())
    refined_voxels = int(refined.sum())
    intersection = int(np.count_nonzero(prompt & refined))
    seed_coverage = intersection / max(seed_voxels, 1)
    refined_overlap = intersection / max(refined_voxels, 1)
    volume_ratio = refined_voxels / max(seed_voxels, 1)
    accepted = bool(
        seed_voxels > 0
        and refined_voxels > 0
        and seed_coverage >= min_seed_coverage
        and refined_overlap >= min_refined_overlap
        and min_volume_ratio <= volume_ratio <= max_volume_ratio
    )
    result = ((anchor & ~prompt) | refined) if accepted else anchor.copy()
    return result, {
        "anchor_voxels": int(anchor.sum()),
        "prompt_voxels": seed_voxels,
        "refined_voxels": refined_voxels,
        "intersection_voxels": intersection,
        "seed_coverage": seed_coverage,
        "refined_overlap": refined_overlap,
        "volume_ratio": volume_ratio,
        "accepted": accepted,
        "output_voxels": int(result.sum()),
    }


def save_like(reference: nib.Nifti1Image, mask: np.ndarray, output: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    result = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp{NIFTI_SUFFIX}")
    nib.save(result, temporary)
    os.replace(temporary, output)


def materialize(args: argparse.Namespace) -> dict[str, object]:
    references = sorted(args.references.glob(f"*{NIFTI_SUFFIX}"))
    if args.expected_count is not None and len(references) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} references, found {len(references)}")
    if not references:
        raise ValueError("no reference images")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for reference_path in references:
        case = case_name(reference_path)
        reference = nib.load(reference_path)
        anchor_image = nib.load(args.anchor / f"{case}{NIFTI_SUFFIX}")
        prompt_image = nib.load(args.prompts / f"{case}{NIFTI_SUFFIX}")
        if not same_geometry(reference, anchor_image) or not same_geometry(reference, prompt_image):
            raise ValueError(f"anchor/prompt geometry mismatch for {case}")
        anchor = np.asarray(anchor_image.dataobj) > 0
        prompt = np.asarray(prompt_image.dataobj) > 0
        refined = instance_union(reference_path, args.instances)
        result, row = gated_replacement(
            anchor,
            prompt,
            refined,
            args.min_seed_coverage,
            args.min_refined_overlap,
            args.min_volume_ratio,
            args.max_volume_ratio,
        )
        row["case"] = case
        rows.append(row)
        save_like(reference, result, args.output / f"{case}{NIFTI_SUFFIX}")
    if len(list(args.output.glob(f"*{NIFTI_SUFFIX}"))) != len(references):
        raise ValueError("replacement output inventory mismatch")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "label-free gated replacement of top-K frozen-anchor prompts",
        "cases": len(references),
        "accepted_cases": sum(bool(row["accepted"]) for row in rows),
        "min_seed_coverage": args.min_seed_coverage,
        "min_refined_overlap": args.min_refined_overlap,
        "min_volume_ratio": args.min_volume_ratio,
        "max_volume_ratio": args.max_volume_ratio,
        "records": rows,
        "labels_read": False,
        "hidden_data_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--min-seed-coverage", type=float, default=0.25)
    parser.add_argument("--min-refined-overlap", type=float, default=0.25)
    parser.add_argument("--min-volume-ratio", type=float, default=0.25)
    parser.add_argument("--max-volume-ratio", type=float, default=4.0)
    args = parser.parse_args()
    for value in (args.min_seed_coverage, args.min_refined_overlap):
        if not 0 <= value <= 1:
            raise ValueError("overlap gates must be in [0,1]")
    if args.min_volume_ratio < 0 or args.max_volume_ratio < args.min_volume_ratio:
        raise ValueError("invalid volume-ratio gates")
    payload = materialize(args)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.audit.with_name(f".{args.audit.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.audit)
    with args.audit.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (
            "case", "anchor_voxels", "prompt_voxels", "refined_voxels",
            "intersection_voxels", "seed_coverage", "refined_overlap",
            "volume_ratio", "accepted", "output_voxels",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in payload["records"])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
