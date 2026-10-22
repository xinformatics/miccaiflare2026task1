#!/usr/bin/env python3
"""Audit saved Codabench archives and emit one deduplicated upload queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


# Hidden-validation submissions use the challenge's FLARE-Val identifiers.
# Public labels use FLARE23Ts identifiers and must never be accepted here.
MEMBER_PATTERN = re.compile(r"^FLARE-Val_[0-9]{4}\.nii\.gz$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(path: Path, expected_count: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"corrupt ZIP member {corrupt}: {path}")
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != expected_count or len(set(names)) != expected_count:
            raise ValueError(
                f"expected {expected_count} unique members, found {len(names)}: {path}"
            )
        invalid = [name for name in names if not MEMBER_PATTERN.fullmatch(name)]
        if invalid:
            raise ValueError(f"invalid root-level member names in {path}: {invalid[:3]}")
        for name in sorted(names):
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            with archive.open(name) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest(), len(names)


def load_rows(path: Path, priority: str) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle, delimiter="\t"))
    # Older plan writers used CSV's default CRLF terminator. A bash ``read``
    # loop retained the terminal CR in its last input field and then appended
    # archive/checksum columns after it, producing a two-record physical
    # representation. Repair it so already-running savers cannot strand valid
    # archives.
    rows: list[dict[str, str]] = []
    for row in parsed:
        continuation = (
            not row.get("rank")
            and str(row.get("tag") or "").endswith(".zip")
            and bool(row.get("family"))
            and not row.get("submission")
            and not row.get("sha256")
        )
        if continuation:
            if not rows or rows[-1].get("submission") or rows[-1].get("sha256"):
                raise ValueError(f"orphan CRLF continuation row in {path}")
            rows[-1]["submission"] = str(row["tag"])
            rows[-1]["sha256"] = str(row["family"])
            continue
        rows.append(row)
    for row in rows:
        row["priority"] = priority
        if priority == "strict":
            row["gate"] = "PROMOTE"
    return rows


def evidence_rank(row: dict[str, str]) -> tuple[float, float, float]:
    """Rank candidates inside one gate by balanced public evidence."""
    try:
        dsc = float(row.get("DSC", ""))
        nsd = float(row.get("NSD", ""))
    except (TypeError, ValueError):
        return (float("-inf"), float("-inf"), float("-inf"))
    return ((dsc + nsd) / 2.0, dsc, nsd)


def dsc_target_rank(row: dict[str, str]) -> tuple[float, float, float]:
    """Rank promoted candidates for the DSC target, with NSD as tie-breaker."""
    try:
        dsc = float(row.get("DSC", ""))
        nsd = float(row.get("NSD", ""))
    except (TypeError, ValueError):
        return (float("-inf"), float("-inf"), float("-inf"))
    return (dsc, nsd, (dsc + nsd) / 2.0)


def submitted_archives(path: Path, submissions_dir: Path) -> tuple[set[str], set[str]]:
    binary: set[str] = set()
    content: set[str] = set()
    if not path.is_file():
        return binary, content
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row or not row[0] or row[0].lower() == "sha256":
                continue
            binary.add(row[0])
            if len(row) > 1:
                archive = submissions_dir / row[1]
                if archive.is_file():
                    try:
                        digest, _ = content_sha256(archive, 100)
                    except (OSError, ValueError, zipfile.BadZipFile):
                        continue
                    content.add(digest)
    return binary, content


def finalize(
    strict: Path,
    backup: Path,
    submitted: Path,
    output: Path,
    audit_output: Path,
    expected_count: int,
    max_count: int,
) -> list[dict[str, object]]:
    # Gate priority is immutable. The strict promotion gate has already
    # required practical improvements in both DSC and NSD, so rank promoted
    # candidates for the explicit DSC target and use NSD as the tie-breaker.
    # Ungated fallbacks retain balanced evidence ranking.
    strict_rows = sorted(
        load_rows(strict, "strict"), key=dsc_target_rank, reverse=True
    )
    backup_rows = sorted(
        load_rows(backup, "backup"), key=evidence_rank, reverse=True
    )
    rows = strict_rows + backup_rows
    submissions_dir = output.parent
    submitted_binary, submitted_content = submitted_archives(
        submitted, submissions_dir
    )
    seen_binary = set(submitted_binary)
    seen_content = set(submitted_content)
    ready: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []

    for row in rows:
        if len(ready) >= max_count:
            rejected.append({"tag": row.get("tag", ""), "reason": "capacity"})
            continue
        archive = Path(row["submission"]).resolve()
        tag = row.get("tag", archive.stem)
        try:
            if len(archive.name) >= 64:
                raise ValueError("archive filename is 64 characters or longer")
            if not archive.is_file() or archive.stat().st_size == 0:
                raise ValueError("archive is missing or empty")
            binary_digest = file_sha256(archive)
            recorded = row.get("sha256", "")
            if recorded and recorded != binary_digest:
                raise ValueError("recorded SHA256 does not match archive")
            content_digest, members = content_sha256(archive, expected_count)
            if binary_digest in seen_binary:
                raise ValueError("binary-equivalent to an existing/submitted archive")
            if content_digest in seen_content:
                raise ValueError("prediction-content-equivalent to an existing/submitted archive")
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            rejected.append({"tag": tag, "reason": str(error)})
            continue
        seen_binary.add(binary_digest)
        seen_content.add(content_digest)
        ready.append(
            {
                "rank": len(ready) + 1,
                "priority": row["priority"],
                "tag": tag,
                "family": row.get("family", ""),
                "DSC": row.get("DSC", ""),
                "NSD": row.get("NSD", ""),
                "gate": row.get("gate", ""),
                "submission": str(archive),
                "sha256": binary_digest,
                "content_sha256": content_digest,
                "member_count": members,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "rank",
        "priority",
        "tag",
        "family",
        "DSC",
        "NSD",
        "gate",
        "submission",
        "sha256",
        "content_sha256",
        "member_count",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(ready)
    audit_output.write_text(
        json.dumps(
            {
                "ready_count": len(ready),
                "candidate_count": len(rows),
                "expected_member_count": expected_count,
                "max_count": max_count,
                "rejected": rejected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--submitted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--max-count", type=int, default=3)
    args = parser.parse_args()
    ready = finalize(
        args.strict,
        args.backup,
        args.submitted,
        args.output,
        args.audit_output,
        args.expected_count,
        args.max_count,
    )
    print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
