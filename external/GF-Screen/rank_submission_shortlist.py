#!/usr/bin/env python3
"""Rank at most three integrity-checked Codabench submission candidates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np


METRICS = ("Lesion_DSC", "Lesion_NSD")
MASK_PATTERN = re.compile(r"FLARE-Val_(\d{4})\.nii\.gz$")


@dataclass
class Candidate:
    name: str
    family: str
    metrics: str
    submission: str
    gate: str
    gate_decision: str
    sha256: str
    mean_dsc: float
    mean_nsd: float
    dsc_gain: float
    nsd_gain: float
    eligible: bool
    status: str
    hidden_total_volume_ratio: float = 1.0
    hidden_median_abs_log_volume_ratio: float = 0.0
    hidden_p90_abs_log_volume_ratio: float = 0.0
    hidden_empty_disagreement_cases: int = 0
    hidden_sanity_pass: bool = True


def resolve_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def read_metrics(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no metric rows")
    required = {"Name", *METRICS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    names = [row["Name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate case names")
    return names, {
        metric: [float(row[metric]) for row in rows] for metric in METRICS
    }


def validate_submission(path: Path, expected_masks: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"{path}: CRC failure in {corrupt}")
        files = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(files) != expected_masks:
            raise ValueError(
                f"{path}: expected {expected_masks} files, found {len(files)}"
            )
        expected_indices = set(range(1, expected_masks + 1))
        observed_indices: set[int] = set()
        for name in files:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1:
                raise ValueError(f"{path}: unsafe or nested archive member {name!r}")
            match = MASK_PATTERN.fullmatch(name)
            if not match:
                raise ValueError(f"{path}: unexpected archive member {name!r}")
            observed_indices.add(int(match.group(1)))
        if observed_indices != expected_indices:
            missing = sorted(expected_indices - observed_indices)
            extra = sorted(observed_indices - expected_indices)
            raise ValueError(f"{path}: mask index mismatch missing={missing} extra={extra}")
    return digest.hexdigest()


def archive_foreground_counts(path: Path) -> list[int]:
    """Count binary foreground without retaining whole CT volumes in RAM.

    Reading a gzipped NIfTI member into ``BytesIO`` temporarily holds the ZIP
    payload, the fully inflated NIfTI, and the NumPy array together.  Ranking a
    portfolio can therefore be killed by memory pressure despite processing
    archives sequentially. Parse the header and count bounded raw-data chunks
    directly from the nested gzip stream instead.
    """

    counts: list[int] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            info.filename for info in archive.infolist() if not info.is_dir()
        )
        for name in names:
            try:
                with archive.open(name) as compressed_member:
                    with gzip.GzipFile(fileobj=compressed_member) as inflated:
                        header = nib.Nifti1Header.from_fileobj(inflated)
                        shape = tuple(int(value) for value in header.get_data_shape())
                        if not shape or any(value < 1 for value in shape):
                            raise ValueError(f"invalid NIfTI shape: {shape}")
                        dtype = np.dtype(header.get_data_dtype())
                        if dtype.itemsize < 1:
                            raise ValueError(f"invalid NIfTI dtype: {dtype}")
                        offset = int(header.get_data_offset())
                        if offset < inflated.tell():
                            raise ValueError(
                                f"data offset {offset} precedes parsed header"
                            )
                        inflated.seek(offset)
                        slope, intercept = header.get_slope_inter()
                        slope = 1.0 if slope is None else float(slope)
                        intercept = 0.0 if intercept is None else float(intercept)
                        remaining = int(np.prod(shape, dtype=np.int64))
                        foreground = 0
                        chunk_elements = max(1, (8 * 1024 * 1024) // dtype.itemsize)
                        while remaining:
                            elements = min(remaining, chunk_elements)
                            raw = inflated.read(elements * dtype.itemsize)
                            if len(raw) != elements * dtype.itemsize:
                                raise ValueError(
                                    f"truncated voxel data: expected {elements * dtype.itemsize} bytes, "
                                    f"read {len(raw)}"
                                )
                            values = np.frombuffer(raw, dtype=dtype, count=elements)
                            if slope != 1.0 or intercept != 0.0:
                                values = values.astype(np.float64) * slope + intercept
                            labels = np.unique(values)
                            if not np.isin(labels, (0, 1)).all():
                                raise ValueError(
                                    f"non-binary labels: {labels[:20].tolist()}"
                                )
                            foreground += int(np.count_nonzero(values))
                            remaining -= elements
                        counts.append(foreground)
            except Exception as error:
                raise ValueError(
                    f"{path}: cannot decode {name}: {error}"
                ) from error
    return counts


def foreground_drift(
    anchor_counts: list[int], candidate_counts: list[int]
) -> dict[str, float | int | bool]:
    if len(anchor_counts) != len(candidate_counts) or not anchor_counts:
        raise ValueError("hidden foreground vectors must be nonempty and aligned")
    anchor = np.asarray(anchor_counts, dtype=np.float64)
    candidate = np.asarray(candidate_counts, dtype=np.float64)
    absolute_log_ratio = np.abs(np.log((candidate + 1.0) / (anchor + 1.0)))
    total_ratio = float((candidate.sum() + 1.0) / (anchor.sum() + 1.0))
    median = float(np.median(absolute_log_ratio))
    p90 = float(np.quantile(absolute_log_ratio, 0.90))
    empty_disagreement = int(np.count_nonzero((anchor == 0) != (candidate == 0)))
    # This is deliberately a broad catastrophe guard, not a performance gate:
    # valid recall-oriented models may differ substantially from the anchor.
    safe = bool(
        0.5 <= total_ratio <= 2.0
        and median <= math.log(2.0)
        and p90 <= math.log(10.0)
        and empty_disagreement <= max(20, len(anchor_counts) // 5)
    )
    return {
        "hidden_total_volume_ratio": total_ratio,
        "hidden_median_abs_log_volume_ratio": median,
        "hidden_p90_abs_log_volume_ratio": p90,
        "hidden_empty_disagreement_cases": empty_disagreement,
        "hidden_sanity_pass": safe,
    }


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_candidates(
    manifest: Path,
    anchor: Path,
    expected_cases: int,
    expected_masks: int,
    hidden_anchor: Path | None = None,
) -> tuple[list[Candidate], dict[str, float]]:
    anchor_names, anchor_values = read_metrics(anchor)
    if len(anchor_names) != expected_cases:
        raise ValueError(
            f"{anchor}: expected {expected_cases} metric rows, found {len(anchor_names)}"
        )
    anchor_means = {
        metric: sum(anchor_values[metric]) / expected_cases for metric in METRICS
    }
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"name", "family", "metrics", "submission", "gate", "submitted"}
    if not rows:
        raise ValueError(f"{manifest}: no candidate rows")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{manifest}: missing columns {sorted(missing)}")

    hidden_anchor_counts = (
        archive_foreground_counts(hidden_anchor) if hidden_anchor is not None else None
    )
    candidates: list[Candidate] = []
    seen_names: set[str] = set()
    for row in rows:
        name = row["name"].strip()
        if not name or name in seen_names:
            raise ValueError(f"{manifest}: empty or duplicate candidate name {name!r}")
        seen_names.add(name)
        metrics_path = resolve_path(row["metrics"], manifest)
        submission_path = resolve_path(row["submission"], manifest)
        gate_path = resolve_path(row["gate"], manifest)
        names, values = read_metrics(metrics_path)
        if names != anchor_names:
            raise ValueError(f"{metrics_path}: case order/names differ from anchor")
        means = {metric: sum(values[metric]) / expected_cases for metric in METRICS}
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        gate_decision = str(gate.get("decision", "MISSING"))
        sha256 = validate_submission(submission_path, expected_masks)
        submitted = truthy(row["submitted"])
        drift = (
            foreground_drift(
                hidden_anchor_counts, archive_foreground_counts(submission_path)
            )
            if hidden_anchor_counts is not None and not submitted
            else foreground_drift([1], [1])
        )
        eligible = (
            not submitted
            and gate_decision == "PROMOTE"
            and means["Lesion_DSC"] >= anchor_means["Lesion_DSC"]
            and means["Lesion_NSD"] >= anchor_means["Lesion_NSD"]
            and bool(drift["hidden_sanity_pass"])
        )
        dsc_nonregress = means["Lesion_DSC"] >= anchor_means["Lesion_DSC"]
        nsd_nonregress = means["Lesion_NSD"] >= anchor_means["Lesion_NSD"]
        if submitted:
            status = "EXCLUDE_PREVIOUSLY_SUBMITTED"
        elif not drift["hidden_sanity_pass"]:
            status = "FALLBACK_HIDDEN_DISTRIBUTION_DRIFT"
        elif not nsd_nonregress:
            status = "FALLBACK_NSD_REGRESSION"
        elif not dsc_nonregress:
            status = "FALLBACK_DSC_REGRESSION"
        elif gate_decision != "PROMOTE":
            status = "FALLBACK_NONREGRESS_GATE_UNCERTAIN"
        else:
            status = "PROMOTED"
        candidates.append(Candidate(
            name=name,
            family=row["family"].strip(),
            metrics=str(metrics_path.resolve()),
            submission=str(submission_path.resolve()),
            gate=str(gate_path.resolve()),
            gate_decision=gate_decision,
            sha256=sha256,
            mean_dsc=means["Lesion_DSC"],
            mean_nsd=means["Lesion_NSD"],
            dsc_gain=means["Lesion_DSC"] - anchor_means["Lesion_DSC"],
            nsd_gain=means["Lesion_NSD"] - anchor_means["Lesion_NSD"],
            eligible=eligible,
            status=status,
            **drift,
        ))
    return candidates, anchor_means


def choose(candidates: list[Candidate], count: int) -> list[Candidate]:
    # A byte-identical archive can consume only one slot regardless of its label.
    unique: list[Candidate] = []
    seen_hashes: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            not item.eligible,
            not item.hidden_sanity_pass,
            item.nsd_gain < 0,
            item.dsc_gain < 0,
            -item.mean_dsc,
            -item.mean_nsd,
            item.name,
        ),
    ):
        if candidate.status == "EXCLUDE_PREVIOUSLY_SUBMITTED":
            continue
        if candidate.sha256 in seen_hashes:
            candidate.status = "EXCLUDE_DUPLICATE_ARCHIVE"
            continue
        seen_hashes.add(candidate.sha256)
        unique.append(candidate)

    selected: list[Candidate] = []

    def safety_tier(item: Candidate) -> tuple[bool, bool, bool, bool]:
        """Keep promoted and non-regressing candidates ahead of risky fallbacks."""
        return (
            not item.eligible,
            not item.hidden_sanity_pass,
            item.nsd_gain < 0,
            item.dsc_gain < 0,
        )

    # Use the scarce leaderboard slots deliberately within each safety tier:
    #   1. highest deployed public DSC;
    #   2. highest NSD among candidates within 0.005 DSC of that leader;
    #   3. strongest remaining candidate from a different model family;
    # then fill by DSC. This keeps DSC as the primary target, preserves a
    # competitive boundary-quality hedge, and avoids spending all slots on
    # effectively the same training hypothesis. No lower safety tier can
    # displace a candidate from a higher one.
    tiers: list[tuple[bool, bool, bool, bool]] = []
    for candidate in unique:
        tier = safety_tier(candidate)
        if tier not in tiers:
            tiers.append(tier)
    for tier in tiers:
        pool = [item for item in unique if safety_tier(item) == tier]
        if not pool:
            continue

        dsc_leader = pool[0]
        selected.append(dsc_leader)
        if len(selected) == count:
            return selected

        near_dsc = [
            item for item in pool
            if item not in selected and item.mean_dsc >= dsc_leader.mean_dsc - 0.005
        ]
        if near_dsc:
            nsd_hedge = max(
                near_dsc,
                key=lambda item: (item.mean_nsd, item.mean_dsc, item.name),
            )
            selected.append(nsd_hedge)
            if len(selected) == count:
                return selected

        used_families = {item.family for item in selected}
        diverse = next(
            (
                item for item in pool
                if item not in selected and item.family not in used_families
            ),
            None,
        )
        if diverse is not None:
            selected.append(diverse)
            if len(selected) == count:
                return selected

        for candidate in pool:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) == count:
                return selected
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--expected-cases", type=int, default=50)
    parser.add_argument("--expected-masks", type=int, default=100)
    parser.add_argument(
        "--hidden-anchor", type=Path,
        help="verified hidden archive used only for label-free volume-drift checks",
    )
    args = parser.parse_args()
    if args.count < 1 or args.expected_cases < 1 or args.expected_masks < 1:
        parser.error("counts must be positive")

    candidates, anchor_means = load_candidates(
        args.manifest, args.anchor, args.expected_cases, args.expected_masks,
        args.hidden_anchor,
    )
    selected = choose(candidates, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking = {
        "anchor": {"path": str(args.anchor.resolve()), **anchor_means},
        "requested_slots": args.count,
        "selected_count": len(selected),
        "selected": [asdict(candidate) for candidate in selected],
        "all_candidates": [asdict(candidate) for candidate in candidates],
    }
    (args.output_dir / "ranking.json").write_text(
        json.dumps(ranking, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "rank", "status", "name", "family", "mean_dsc", "mean_nsd",
        "dsc_gain", "nsd_gain", "hidden_sanity_pass",
        "hidden_total_volume_ratio", "hidden_median_abs_log_volume_ratio",
        "hidden_p90_abs_log_volume_ratio", "hidden_empty_disagreement_cases",
        "sha256", "submission", "metrics", "gate",
    ]
    with (args.output_dir / "top3.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for rank, candidate in enumerate(selected, 1):
            row = asdict(candidate)
            row["rank"] = rank
            writer.writerow({column: row[column] for column in columns})
    print(f"selected {len(selected)}/{args.count} unique candidates")
    for rank, candidate in enumerate(selected, 1):
        print(
            f"{rank}. {candidate.name} [{candidate.status}] "
            f"DSC={candidate.mean_dsc:.6f} NSD={candidate.mean_nsd:.6f} "
            f"{candidate.submission}"
        )


if __name__ == "__main__":
    main()
