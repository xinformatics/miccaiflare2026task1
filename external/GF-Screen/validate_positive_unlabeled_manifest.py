#!/usr/bin/env python3
"""Prove that a pseudo-positive manifest obeys the Train-Unlabel contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_paths(values: list[str]) -> set[str]:
    return {str(Path(value).resolve()) for value in values}


def source_cohort(path: str, root: Path) -> str:
    relative = Path(path).resolve().relative_to(root.resolve())
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


def validate_contract(
    pseudo_terminal_path: Path,
    manifest_path: Path,
    pseudo_audit_path: Path,
) -> dict[str, object]:
    terminal = json.loads(pseudo_terminal_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pseudo_audit = json.loads(pseudo_audit_path.read_text(encoding="utf-8"))
    if terminal.get("status") != "COMPLETE" or terminal.get("decision") != (
        "PSEUDO_LABELS_READY_FOR_LONG_TRAINING"
    ):
        raise ValueError("pseudo-label terminal is not complete")

    train_identity = terminal.get("train_unlabel_audit")
    if not isinstance(train_identity, dict) or not train_identity.get("path"):
        raise ValueError("terminal lacks Train-Unlabel audit identity")
    train_audit_path = Path(train_identity["path"])
    if sha256(train_audit_path) != train_identity.get("sha256"):
        raise ValueError("Train-Unlabel audit identity hash mismatch")
    train_audit = json.loads(train_audit_path.read_text(encoding="utf-8"))
    if train_audit.get("status") not in {"PASS", "PASS_WITH_EXCLUSIONS"}:
        raise ValueError("Train-Unlabel source audit did not pass")

    pseudo_identity = terminal.get("balanced_pseudo_audit")
    if not isinstance(pseudo_identity, dict):
        raise ValueError("terminal lacks balanced pseudo-audit identity")
    if Path(pseudo_identity.get("path", "")).resolve() != pseudo_audit_path.resolve():
        raise ValueError("terminal points to a different balanced pseudo audit")
    if sha256(pseudo_audit_path) != pseudo_identity.get("sha256"):
        raise ValueError("balanced pseudo-audit identity hash mismatch")

    unlabeled_root = Path(train_audit["unlabeled_root"])
    source_images = {
        str(path.resolve())
        for path in unlabeled_root.rglob("*.nii.gz")
        if path.is_file()
    }
    excluded = resolved_paths(train_audit.get("excluded_unlabeled_files", []))
    if not excluded <= source_images:
        raise ValueError("Train-Unlabel exclusions are outside the source inventory")
    usable = source_images - excluded
    if len(source_images) != int(train_audit.get("unlabeled_files", -1)):
        raise ValueError("live Train-Unlabel inventory differs from its audit")
    if len(usable) != int(train_audit.get("usable_unique_files", -1)):
        raise ValueError("usable Train-Unlabel inventory differs from its audit")

    if int(pseudo_audit.get("images", -1)) != len(usable):
        raise ValueError("pseudo audit does not cover every usable source image")
    if int(pseudo_audit.get("excluded_exact_overlaps", -1)) != len(excluded):
        raise ValueError("pseudo audit exclusion count mismatch")
    if resolved_paths(pseudo_audit.get("excluded_unlabeled_files", [])) != excluded:
        raise ValueError("pseudo audit exclusion identity mismatch")

    cases = pseudo_audit.get("cases")
    if not isinstance(cases, list) or len(cases) != len(usable):
        raise ValueError("pseudo audit case inventory is incomplete")
    case_images = [str(Path(case.get("image", "")).resolve()) for case in cases]
    if len(case_images) != len(set(case_images)) or set(case_images) != usable:
        raise ValueError("pseudo audit cases are not exactly the usable source set")

    training = manifest.get("training")
    if not isinstance(training, list) or not training:
        raise ValueError("positive-only pseudo manifest is empty")
    if any(item.get("pseudo_label") is not True for item in training):
        raise ValueError("pseudo manifest contains non-positive-only entries")
    training_images = [
        str(Path(item.get("image", "")).resolve()) for item in training
    ]
    if len(training_images) != len(set(training_images)):
        raise ValueError("pseudo manifest contains duplicate source images")
    if not set(training_images) <= usable or set(training_images) & excluded:
        raise ValueError("pseudo manifest includes a non-usable or excluded image")

    positive_cases = [case for case in cases if int(case.get("components", 0)) > 0]
    positive_images = {str(Path(case["image"]).resolve()) for case in positive_cases}
    if set(training_images) != positive_images:
        raise ValueError("manifest entries do not exactly match positive audit cases")
    if len(training) != int(pseudo_audit.get("positive_cases", -1)):
        raise ValueError("pseudo manifest/audit positive count mismatch")
    if int(pseudo_audit.get("empty_or_rejected_cases", -1)) + len(training) != len(
        usable
    ):
        raise ValueError("pseudo positive/empty inventory mismatch")

    source_cohort_counts = Counter(source_cohort(path, unlabeled_root) for path in usable)
    positive_cohort_counts = Counter(
        source_cohort(path, unlabeled_root) for path in training_images
    )
    missing_positive_cohorts = sorted(
        cohort for cohort in source_cohort_counts if positive_cohort_counts[cohort] == 0
    )
    if missing_positive_cohorts:
        raise ValueError(
            "source cohorts contain no positive pseudo cases: "
            + ", ".join(missing_positive_cohorts)
        )
    empty_cohort_counts = {
        cohort: source_cohort_counts[cohort] - positive_cohort_counts[cohort]
        for cohort in sorted(source_cohort_counts)
    }

    audit_labels = {
        str(Path(case["image"]).resolve()): {
            "path": str(Path(case["label"]).resolve()),
            "sha256": str(case.get("label_sha256", "")),
            "bytes": int(case.get("label_bytes", -1)),
        }
        for case in positive_cases
    }
    for item in training:
        image = str(Path(item["image"]).resolve())
        label = Path(item.get("label", ""))
        identity = audit_labels[image]
        if str(label.resolve()) != identity["path"] or not label.is_file():
            raise ValueError(f"manifest label identity mismatch for {image}")
        if (
            len(identity["sha256"]) != 64
            or identity["bytes"] < 1
            or str(item.get("label_sha256", "")) != identity["sha256"]
            or int(item.get("label_bytes", -1)) != identity["bytes"]
        ):
            raise ValueError(f"manifest label digest metadata mismatch for {image}")
        if label.stat().st_size != identity["bytes"] or sha256(label) != identity["sha256"]:
            raise ValueError(f"pseudo-label content hash mismatch for {image}")

    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "source_images": len(source_images),
        "excluded_images": len(excluded),
        "usable_images": len(usable),
        "positive_cases": len(training),
        "empty_or_rejected_cases": len(usable) - len(training),
        "source_cohort_counts": dict(sorted(source_cohort_counts.items())),
        "positive_cohort_counts": dict(sorted(positive_cohort_counts.items())),
        "empty_or_rejected_cohort_counts": empty_cohort_counts,
        "excluded_unlabeled_files": sorted(excluded),
        "train_unlabel_audit": {
            "path": str(train_audit_path.resolve()),
            "sha256": sha256(train_audit_path),
        },
        "pseudo_terminal": {
            "path": str(pseudo_terminal_path.resolve()),
            "sha256": sha256(pseudo_terminal_path),
        },
        "pseudo_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "pseudo_audit": {
            "path": str(pseudo_audit_path.resolve()),
            "sha256": sha256(pseudo_audit_path),
        },
        "contract": (
            "manifest is exactly the positive subset of all usable Train-Unlabel "
            "images; excluded overlaps and teacher-empty cases are absent; "
            "every source cohort contributes at least one positive case"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo-terminal", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pseudo-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_contract(
        args.pseudo_terminal, args.manifest, args.pseudo_audit
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
