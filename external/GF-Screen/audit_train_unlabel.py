#!/usr/bin/env python3
"""Audit Train-Unlabel identity, readability, and challenge-data separation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib


def nifti_files(root: Path):
    return sorted(path for path in root.rglob("*.nii.gz") if path.is_file())


def canonical_labeled_key(path: Path):
    name = path.name.removesuffix("_0000.nii.gz")
    patterns = (
        ("msd_pancreas", r"^Pancreas-MSD_(\d+)$"),
        ("msd_lung", r"^Chest_MSD-LungTumor_(\d+)$"),
        ("msd_hepatic", r"^Liver-MSD-HepaticVessel_(\d+)$"),
        ("msd_liver", r"^Liver-MSD_(\d+)$"),
        ("msd_colon", r"^Colon-MSD_(\d+)$"),
    )
    for source, pattern in patterns:
        match = re.match(pattern, name)
        if match:
            return f"{source}:{int(match.group(1))}"
    return None


def canonical_unlabeled_key(path: Path):
    source = {
        "MSD-Pancreas": "msd_pancreas",
        "MSD-Lung": "msd_lung",
        "MSD-HapaticVessel": "msd_hepatic",
        "MSD-Liver": "msd_liver",
        "MSD-Colon": "msd_colon",
        "MSD-Spleen": "msd_spleen",
    }.get(path.parent.name)
    if source is None:
        return None
    match = re.search(r"(\d+)$", path.name.removesuffix(".nii.gz"))
    if match is None:
        raise ValueError(f"cannot parse MSD case id: {path}")
    return f"{source}:{int(match.group(1))}"


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled", type=Path, required=True)
    parser.add_argument("--labeled", type=Path, action="append", default=[])
    parser.add_argument("--validation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    unlabeled = nifti_files(args.unlabeled)
    labeled = [path for root in args.labeled for path in nifti_files(root)]
    validation = [path for root in args.validation for path in nifti_files(root)]
    basenames = Counter(path.name for path in unlabeled)
    duplicate_basenames = sorted(name for name, count in basenames.items() if count > 1)

    labeled_keys = {
        key for path in labeled if (key := canonical_labeled_key(path)) is not None
    }
    unlabeled_keys = {
        key for path in unlabeled if (key := canonical_unlabeled_key(path)) is not None
    }
    canonical_overlap = sorted(labeled_keys & unlabeled_keys)

    # Exact compressed-file duplicates are cheap to rule out by hashing only
    # equal-size candidates. Source canonical IDs additionally cover MSD files
    # that may have been recompressed or renamed.
    reference_by_size = defaultdict(list)
    for path in labeled + validation:
        reference_by_size[path.stat().st_size].append(path)
    exact_duplicates = []
    reference_hash_cache = {}
    for path in unlabeled:
        candidates = reference_by_size.get(path.stat().st_size, ())
        if not candidates:
            continue
        digest = file_sha256(path)
        for candidate in candidates:
            candidate_digest = reference_hash_cache.setdefault(
                candidate, file_sha256(candidate)
            )
            if digest == candidate_digest:
                exact_duplicates.append(
                    {
                        "unlabeled": str(path.resolve()),
                        "reference": str(candidate.resolve()),
                    }
                )
    excluded_unlabeled = sorted(
        {item["unlabeled"] for item in exact_duplicates}
    )

    unreadable = []
    shapes = Counter()
    for path in unlabeled:
        try:
            image = nib.load(path)
            shapes[str(tuple(int(value) for value in image.shape))] += 1
            if len(image.shape) != 3:
                unreadable.append({"path": str(path), "error": f"shape={image.shape}"})
        except Exception as error:  # pragma: no cover - data-dependent
            unreadable.append({"path": str(path), "error": repr(error)})

    cohort_counts = Counter(
        path.relative_to(args.unlabeled).parts[0] for path in unlabeled
    )
    payload = {
        "schema": "train-unlabel-audit-v1",
        "unlabeled_root": str(args.unlabeled.resolve()),
        "unlabeled_files": len(unlabeled),
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "duplicate_basenames": duplicate_basenames,
        "canonical_msd_overlap_with_train_labeled": canonical_overlap,
        "exact_compressed_overlap_with_labeled_or_validation": exact_duplicates,
        "excluded_unlabeled_files": excluded_unlabeled,
        "usable_unique_files": len(unlabeled) - len(excluded_unlabeled),
        "unreadable_or_non_3d": unreadable,
        "shape_count": len(shapes),
        "status": (
            "FAIL"
            if duplicate_basenames or canonical_overlap or unreadable
            else "PASS_WITH_EXCLUSIONS"
            if exact_duplicates
            else "PASS"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: payload[key] for key in (
        "unlabeled_files", "cohort_counts", "duplicate_basenames",
        "canonical_msd_overlap_with_train_labeled",
        "exact_compressed_overlap_with_labeled_or_validation",
        "excluded_unlabeled_files", "usable_unique_files",
        "unreadable_or_non_3d", "status",
    )}, indent=2))
    if payload["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
