#!/usr/bin/env python3
"""Rebuild the upload queue from every saved-candidate manifest.

This is deliberately save-only: it validates already materialized archives,
deduplicates both ZIP bytes and prediction contents, excludes recorded
Codabench submissions, and atomically replaces READY_20260822.tsv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from finalize_saved_submission_queue import finalize, load_rows


FIELDS = (
    "rank",
    "tag",
    "family",
    "checkpoint",
    "glance_threshold",
    "lesion_threshold",
    "DSC",
    "NSD",
    "metrics",
    "robust",
    "gate",
    "submission",
    "sha256",
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for rank, row in enumerate(rows, 1):
            normalized = {field: row.get(field, "") for field in FIELDS}
            normalized["rank"] = str(rank)
            writer.writerow(normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument(
        "--manifest-root",
        action="append",
        default=[],
        help=(
            "recursively discover saved*submission*.tsv manifests below this "
            "directory; repeatable and deduplicated with --manifest"
        ),
    )
    parser.add_argument("--submitted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--max-count", type=int, default=3)
    args = parser.parse_args()

    manifest_paths = {Path(raw_path).resolve() for raw_path in args.manifest}
    for raw_root in args.manifest_root:
        root = Path(raw_root)
        if root.is_dir():
            manifest_paths.update(
                path.resolve()
                for path in root.rglob("saved*submission*.tsv")
                if path.is_file()
            )
    rows: list[dict[str, str]] = []
    loaded_manifests = []
    for path in sorted(manifest_paths):
        if path.is_file() and path.stat().st_size:
            loaded = load_rows(path, "candidate")
            rows.extend(loaded)
            loaded_manifests.append({"path": str(path), "rows": len(loaded)})

    strict = [row for row in rows if row.get("gate") == "PROMOTE"]
    backup = [row for row in rows if row.get("gate") != "PROMOTE"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="refresh_saved_", dir=args.output.parent
    ) as temporary:
        temporary_path = Path(temporary)
        strict_path = temporary_path / "strict.tsv"
        backup_path = temporary_path / "backup.tsv"
        output_path = temporary_path / "READY.tsv"
        audit_path = temporary_path / "READY.audit.json"
        write_manifest(strict_path, strict)
        write_manifest(backup_path, backup)
        ready = finalize(
            strict_path,
            backup_path,
            args.submitted,
            output_path,
            audit_path,
            args.expected_count,
            args.max_count,
        )
        os.replace(output_path, args.output)
        os.replace(audit_path, args.audit_output)

    audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
    audit.update(
        {
            "manifest_count": len(loaded_manifests),
            "manifest_rows": len(rows),
            "manifests": loaded_manifests,
        }
    )
    temporary_audit = args.audit_output.with_name(
        f".{args.audit_output.name}.{os.getpid()}.tmp"
    )
    temporary_audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_audit, args.audit_output)

    for row in ready:
        print(
            f'{row["rank"]}\t{row["priority"]}\t{row["tag"]}\t'
            f'{row["DSC"]}\t{row["NSD"]}\t{row["submission"]}'
        )


if __name__ == "__main__":
    main()
