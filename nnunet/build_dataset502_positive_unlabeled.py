#!/usr/bin/env python3
"""Build an isolated nnU-Net dataset for positive-only Train-Unlabel tuning.

Dataset 501 is never modified.  Its 17,575 raw symlinks and preprocessed case
triplets are reused by symlink, while the audited 2,311 positive Train-Unlabel
image/teacher-mask pairs receive collision-free identifiers.  A later resumable
preprocessing pass therefore processes only the new pseudo cases.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA_VERSION = 1
TARGET_NAME = "Dataset502_PanCancerPositiveUnlabeled"
CACHE_IDENTIFIER = "nnUNetPlans_3d_fullres_S4D2W32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitized(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9-]+", "-", value).strip("-")
    if not rendered:
        raise ValueError(f"identifier token is empty after sanitizing: {value!r}")
    return rendered


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_symlink(link: Path, source: Path) -> None:
    source = source.resolve()
    if not source.is_file() or source.stat().st_size < 1:
        raise ValueError(f"symlink source is missing: {source}")
    if link.is_symlink():
        if link.resolve() != source:
            raise ValueError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise ValueError(f"refusing to replace existing non-symlink: {link}")
    link.symlink_to(source)


def base_inventory(raw: Path, cache: Path) -> list[str]:
    images = {
        path.name.removesuffix("_0000.nii.gz"): path
        for path in (raw / "imagesTr").glob("*_0000.nii.gz")
    }
    labels = {
        path.name.removesuffix(".nii.gz"): path
        for path in (raw / "labelsTr").glob("*.nii.gz")
    }
    if len(images) != 17_575 or images.keys() != labels.keys():
        raise ValueError("Dataset 501 raw inventory is not the immutable 17,575 cases")
    for identifier in images:
        expected = (
            cache / f"{identifier}.b2nd",
            cache / f"{identifier}_seg.b2nd",
            cache / f"{identifier}.pkl",
        )
        if not all(path.is_file() and path.stat().st_size > 0 for path in expected):
            raise ValueError(f"Dataset 501 preprocessed triplet is incomplete: {identifier}")
    return sorted(images)


def pseudo_inventory(manifest_path: Path, contract_path: Path, train_unlabel: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("status") != "PASS"
        or int(contract.get("positive_cases", -1)) != 2_311
        or Path(contract.get("pseudo_manifest", {}).get("path", "")).resolve()
        != manifest_path.resolve()
        or contract.get("pseudo_manifest", {}).get("sha256") != sha256(manifest_path)
    ):
        raise ValueError("positive-only Train-Unlabel contract is invalid")
    rows = manifest.get("training")
    if not isinstance(rows, list) or len(rows) != 2_311:
        raise ValueError("pseudo manifest is not exactly 2,311 positive cases")
    seen_images: set[Path] = set()
    cohort_indices: defaultdict[str, int] = defaultdict(int)
    result = []
    for row in sorted(rows, key=lambda item: str(Path(item["image"]).resolve())):
        image = Path(row["image"]).resolve()
        label = Path(row["label"]).resolve()
        if image in seen_images or not image.is_relative_to(train_unlabel.resolve()):
            raise ValueError(f"duplicate or out-of-scope pseudo image: {image}")
        seen_images.add(image)
        if not label.is_file() or label.stat().st_size != int(row.get("label_bytes", -1)):
            raise ValueError(f"pseudo label size mismatch: {label}")
        if sha256(label) != row.get("label_sha256"):
            raise ValueError(f"pseudo label hash mismatch: {label}")
        cohort = image.relative_to(train_unlabel.resolve()).parts[0]
        cohort_indices[cohort] += 1
        identifier = f"Pseudo-{sanitized(cohort)}_{cohort_indices[cohort]:04d}"
        result.append(
            {
                "identifier": identifier,
                "cohort": cohort,
                "image": image,
                "label": label,
                "label_sha256": row["label_sha256"],
            }
        )
    if len({row["identifier"] for row in result}) != len(result):
        raise ValueError("generated pseudo identifiers are not unique")
    return result


def materialize(
    *,
    base_raw: Path,
    base_preprocessed: Path,
    target_raw: Path,
    target_preprocessed: Path,
    base_ids: list[str],
    pseudo_rows: list[dict],
    manifest_path: Path,
    contract_path: Path,
) -> dict:
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
    for row in pseudo_rows:
        identifier = row["identifier"]
        ensure_symlink(target_images / f"{identifier}_0000.nii.gz", row["image"])
        ensure_symlink(target_labels / f"{identifier}.nii.gz", row["label"])

    total = len(base_ids) + len(pseudo_rows)
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": total,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    atomic_json(target_raw / "dataset.json", dataset)
    atomic_json(target_preprocessed / "dataset.json", dataset)
    # A plans symlink is unsafe here: nnU-Net resolves the preprocessed and
    # results folders from ``dataset_name`` stored inside the plan. Reusing
    # Dataset501's file byte-for-byte would silently send Dataset502 training
    # back to Dataset501 and exclude every pseudo-positive case. Keep the
    # numerical plan identical, but bind a local copy to the new dataset.
    base_plans_path = base_preprocessed / "nnUNetPlans.json"
    target_plans = json.loads(base_plans_path.read_text(encoding="utf-8"))
    if target_plans.get("dataset_name") != "Dataset501_PanCancer":
        raise ValueError("Dataset 501 plan identity changed")
    target_plans["dataset_name"] = TARGET_NAME
    atomic_json(target_preprocessed / "nnUNetPlans.json", target_plans)
    # The fingerprint contains only aggregate planning statistics and no
    # dataset identity, so sharing it remains both exact and space-efficient.
    ensure_symlink(
        target_preprocessed / "dataset_fingerprint.json",
        base_preprocessed / "dataset_fingerprint.json",
    )

    mapping = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "target_dataset": TARGET_NAME,
        "labeled_cases": len(base_ids),
        "positive_unlabeled_cases": len(pseudo_rows),
        "total_cases": total,
        "pseudo_cohorts": dict(sorted(Counter(row["cohort"] for row in pseudo_rows).items())),
        "pseudo_cases": [
            {
                "identifier": row["identifier"],
                "cohort": row["cohort"],
                "image": str(row["image"]),
                "label": str(row["label"]),
                "label_sha256": row["label_sha256"],
            }
            for row in pseudo_rows
        ],
        "pseudo_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "pseudo_contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)},
        "base_cache_reuse": "symlinked immutable Dataset501 triplets",
        "plans_reuse": "local Dataset502-bound copy of immutable Dataset501 numerical plan",
        "pseudo_cache_state": "raw symlinks ready; 2,311 cases require resumable preprocessing",
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(target_preprocessed / "DATASET_BUILD_COMPLETE.json", mapping)
    return mapping


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base_raw = root / "nnUNet_raw/Dataset501_PanCancerCT"
    base_preprocessed = root / "nnUNet_preprocessed/Dataset501_PanCancer"
    target_raw = root / f"nnUNet_raw/{TARGET_NAME}"
    target_preprocessed = root / f"nnUNet_preprocessed/{TARGET_NAME}"
    manifest = root / "external/GF-Screen/public_sweeps/gf_train_unlabel_cbg_direct_20260825/balanced/manifest.json"
    contract = root / "external/GF-Screen/public_sweeps/gf_focus_pu_long30_e283_to313_20260825/PSEUDO_DATASET_CONTRACT.json"
    train_unlabel = root / "train_unlabel"
    base_ids = base_inventory(base_raw, base_preprocessed / CACHE_IDENTIFIER)
    pseudo_rows = pseudo_inventory(manifest, contract, train_unlabel)
    dry_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "VALIDATED_DRY_RUN" if not args.materialize else "READY_TO_MATERIALIZE",
        "target_dataset": TARGET_NAME,
        "labeled_cases": len(base_ids),
        "positive_unlabeled_cases": len(pseudo_rows),
        "total_cases": len(base_ids) + len(pseudo_rows),
        "pseudo_cohorts": dict(sorted(Counter(row["cohort"] for row in pseudo_rows).items())),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    payload = (
        materialize(
            base_raw=base_raw,
            base_preprocessed=base_preprocessed,
            target_raw=target_raw,
            target_preprocessed=target_preprocessed,
            base_ids=base_ids,
            pseudo_rows=pseudo_rows,
            manifest_path=manifest,
            contract_path=contract,
        )
        if args.materialize
        else dry_payload
    )
    if args.output:
        atomic_json(args.output, payload)
    print(json.dumps(payload if not args.materialize else {key: payload[key] for key in dry_payload}, indent=2))


if __name__ == "__main__":
    main()
