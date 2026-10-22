#!/usr/bin/env python3
"""Select independently qualified Focus and Glance sources for composition.

Each upstream save-only stage writes a one-row manifest only when its public
candidate passes the configured robustness/queue gate.  Missing manifests are
therefore an expected rejection state and fall back to the immutable base for
that module.  Existing manifests are validated before their checkpoint is
allowed to seed a cross-module audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


def nonempty_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing {description}: {path}")
    return path.resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_saved_source(path: Path, base: Path, role: str) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        return {
            "role": role,
            "status": "fallback_base_no_qualified_manifest",
            "checkpoint": str(base),
            "glance_threshold": 0.5,
            "lesion_threshold": 0.5,
            "manifest": str(path.resolve()),
        }

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected exactly one saved row in {path}, found {len(rows)}")
    row = rows[0]
    required_columns = {
        "tag",
        "family",
        "checkpoint",
        "DSC",
        "NSD",
        "metrics",
        "robust",
        "gate",
        "submission",
        "sha256",
    }
    missing_columns = sorted(required_columns - set(row))
    if missing_columns:
        raise ValueError(f"missing columns in {path}: {missing_columns}")

    checkpoint = nonempty_file(Path(row["checkpoint"]), "checkpoint")
    metrics = nonempty_file(Path(row["metrics"]), "metrics")
    robust_path = nonempty_file(Path(row["robust"]), "robustness report")
    submission = nonempty_file(Path(row["submission"]), "submission archive")
    robust = json.loads(robust_path.read_text(encoding="utf-8"))
    if int(robust.get("cases", 0)) != 50:
        raise ValueError(f"robustness report does not cover 50 cases: {robust_path}")
    if robust.get("decision") != row["gate"]:
        raise ValueError(
            f"manifest/robust gate mismatch in {path}: "
            f"{row['gate']} != {robust.get('decision')}"
        )
    digest = row["sha256"].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"invalid SHA-256 in {path}: {row['sha256']!r}")
    observed_digest = file_sha256(submission)
    if observed_digest != digest:
        raise ValueError(
            f"submission SHA-256 mismatch in {path}: {digest} != {observed_digest}"
        )
    with zipfile.ZipFile(submission) as archive:
        bad_member = archive.testzip()
        members = [member for member in archive.namelist() if member and not member.endswith("/")]
    if bad_member is not None:
        raise ValueError(f"corrupt ZIP member in {submission}: {bad_member}")
    if len(members) != 100:
        raise ValueError(
            f"saved submission does not contain 100 files: {submission} ({len(members)})"
        )

    return {
        "role": role,
        "status": "qualified_saved_source",
        "tag": row["tag"],
        "family": row["family"],
        "checkpoint": str(checkpoint),
        "DSC": float(row["DSC"]),
        "NSD": float(row["NSD"]),
        "glance_threshold": float(row["glance_threshold"]),
        "lesion_threshold": float(row["lesion_threshold"]),
        "gate": row["gate"],
        "metrics": str(metrics),
        "robust": str(robust_path),
        "submission": str(submission),
        "submission_sha256": digest,
        "submission_member_count": len(members),
        "manifest": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--focus-manifest", type=Path, required=True)
    parser.add_argument("--glance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = nonempty_file(args.base, "immutable base checkpoint")
    report = {
        "selection_rule": (
            "use each independently save-qualified module checkpoint; "
            "fall back to the immutable base when its save manifest is absent"
        ),
        "base_checkpoint": str(base),
        "focus": load_saved_source(args.focus_manifest, base, "focus"),
        "glance": load_saved_source(args.glance_manifest, base, "glance"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
