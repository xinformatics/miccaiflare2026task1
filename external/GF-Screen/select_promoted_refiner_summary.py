#!/usr/bin/env python3
"""Select one strictly promoted candidate-refiner public result.

This utility is deliberately label-agnostic. It validates an already scored
public summary and its robust-decision sidecars, then emits one deterministic
TSV record for a save-only hidden consumer. No eligible row produces no output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = (
    "variant",
    "classification_threshold",
    "segmentation_threshold",
    "deployment_mode",
    "deployment",
    "DSC",
    "NSD",
    "combined",
    "metrics",
    "robust",
    "decision",
)
MODES = {"refined", "refined_or_seed", "seed", "seed_union"}
DEPLOYMENTS = {"anchor", "standalone"}


def resolve_sidecar(summary: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = summary.parent / path
    return local if local.is_file() else path


def select_promoted(
    summary: Path,
    expected_cases: int = 50,
    allow_anchor_fallback: bool = False,
) -> dict[str, str] | None:
    if expected_cases < 1:
        raise ValueError("expected_cases must be positive")
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("public summary is empty")
    missing = set(FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"public summary is missing columns: {sorted(missing)}")

    eligible: list[dict[str, str]] = []
    anchor_fallbacks: list[dict[str, str]] = []
    variants: set[str] = set()
    for row in rows:
        variant = row["variant"].strip()
        if not variant or variant in variants:
            raise ValueError(f"missing or duplicate variant: {variant!r}")
        variants.add(variant)
        if row["deployment_mode"] not in MODES:
            raise ValueError(f"invalid deployment mode for {variant}")
        if row["deployment"] not in DEPLOYMENTS:
            raise ValueError(f"invalid deployment for {variant}")
        classification = float(row["classification_threshold"])
        segmentation = float(row["segmentation_threshold"])
        dsc = float(row["DSC"])
        nsd = float(row["NSD"])
        combined = float(row["combined"])
        if not 0 < classification < 1 or not 0 < segmentation < 1:
            raise ValueError(f"invalid probability threshold for {variant}")
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (dsc, nsd, combined)):
            raise ValueError(f"invalid metric for {variant}")
        if not math.isclose(combined, (dsc + nsd) / 2, abs_tol=5e-4):
            raise ValueError(f"combined metric mismatch for {variant}")
        for key in ("metrics", "robust"):
            if not resolve_sidecar(summary, row[key]).is_file():
                raise ValueError(f"missing {key} sidecar for {variant}: {row[key]}")
        metrics_path = resolve_sidecar(summary, row["metrics"])
        with metrics_path.open(newline="", encoding="utf-8") as stream:
            metric_rows = list(csv.DictReader(stream))
        metric_names = [str(metric_row.get("Name", "")) for metric_row in metric_rows]
        if (
            len(metric_rows) != expected_cases
            or any(not name for name in metric_names)
            or len(metric_names) != len(set(metric_names))
        ):
            raise ValueError(
                f"metrics coverage mismatch for {variant}: "
                f"expected={expected_cases} observed={len(metric_rows)}"
            )
        robust_path = resolve_sidecar(summary, row["robust"])
        robust = json.loads(robust_path.read_text(encoding="utf-8"))
        if int(robust.get("cases", -1)) != expected_cases:
            raise ValueError(f"robust case count mismatch for {variant}")
        candidate_mean = robust.get("candidate_mean", {})
        if not math.isclose(float(candidate_mean.get("Lesion_DSC", -1)), dsc, abs_tol=5e-4):
            raise ValueError(f"robust DSC mismatch for {variant}")
        if not math.isclose(float(candidate_mean.get("Lesion_NSD", -1)), nsd, abs_tol=5e-4):
            raise ValueError(f"robust NSD mismatch for {variant}")
        robust_decision = str(robust.get("decision", ""))
        if row["decision"] != robust_decision:
            raise ValueError(
                f"decision mismatch for {variant}: summary={row['decision']} "
                f"robust={robust_decision}"
            )
        if robust_decision == "PROMOTE":
            checks = robust.get("checks", {})
            if not checks or not all(value is True for value in checks.values()):
                raise ValueError(f"PROMOTE has incomplete or failed checks for {variant}")
            eligible.append(row)
        elif (
            allow_anchor_fallback
            and robust_decision == "KEEP_ANCHOR"
            and row["deployment"] == "anchor"
        ):
            # The caller has already enforced the independent OOF qualification
            # gate. Preserve the champion-anchored distribution-shift candidate
            # even when the small fixed-public set does not prove superiority.
            # It remains marked KEEP_ANCHOR and is never treated as promoted.
            anchor_fallbacks.append(row)

    candidates = eligible or anchor_fallbacks
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (float(row["DSC"]), float(row["NSD"]), row["variant"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=50)
    parser.add_argument(
        "--allow-anchor-fallback",
        action="store_true",
        help=(
            "When no strict PROMOTE exists, emit the best anchor deployment "
            "that is explicitly marked KEEP_ANCHOR. The caller must have "
            "independently enforced OOF qualification."
        ),
    )
    args = parser.parse_args()
    selected = select_promoted(
        args.summary,
        args.expected_cases,
        allow_anchor_fallback=args.allow_anchor_fallback,
    )
    if selected is not None:
        print("\t".join(str(selected[field]) for field in FIELDS))


if __name__ == "__main__":
    main()
