#!/usr/bin/env python3
"""Gate prompt-masked classifier pooling on one matched grouped fold.

The intervention is intended to improve classification of a candidate in its
prompted spatial context.  Promotion therefore requires gains on the two
measured weak proposal families, not just a pooled metric, while preserving
the mature-source classifier and the shared segmentation decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

from select_candidate_refiner_source_conditioning import summarize


REQUIRED_GROUPS = ("mature", "weak_sources", "lung_cells")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_matched_audits(baseline: dict, candidate: dict) -> None:
    identity_keys = ("fold", "folds", "seed", "index_sha256")
    if any(baseline.get(key) != candidate.get(key) for key in identity_keys):
        raise ValueError("source audits do not share fold/index identity")
    for name, audit in (("baseline", baseline), ("candidate", candidate)):
        groups = audit.get("source_classification_groups", {})
        missing = set(REQUIRED_GROUPS) - set(groups)
        if missing:
            raise ValueError(f"{name} source audit misses groups: {sorted(missing)}")
        for group in REQUIRED_GROUPS:
            value = float(groups[group]["average_precision"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {name} {group} average precision")


def group_ap(audit: dict, group: str) -> float:
    return float(audit["source_classification_groups"][group]["average_precision"])


def decide(
    baseline_summary: dict[str, dict[str, float | int]],
    candidate_summary: dict[str, dict[str, float | int]],
    baseline_audit: dict,
    candidate_audit: dict,
    minimum_classification_gain: float = 0.005,
    minimum_weak_source_gain: float = 0.010,
    minimum_lung_cell_gain: float = 0.005,
    maximum_mature_source_drop: float = 0.005,
    maximum_segmentation_drop: float = 0.005,
    minimum_composite_gain: float = 0.0,
) -> tuple[str, dict[str, float], dict[str, bool]]:
    validate_matched_audits(baseline_audit, candidate_audit)
    gains = {
        "average_precision": float(
            candidate_summary["classification"]["average_precision"]
        )
        - float(baseline_summary["classification"]["average_precision"]),
        "weak_sources_average_precision": group_ap(candidate_audit, "weak_sources")
        - group_ap(baseline_audit, "weak_sources"),
        "lung_cells_average_precision": group_ap(candidate_audit, "lung_cells")
        - group_ap(baseline_audit, "lung_cells"),
        "mature_average_precision": group_ap(candidate_audit, "mature")
        - group_ap(baseline_audit, "mature"),
        "segmentation_dice": float(
            candidate_summary["segmentation"]["segmentation_dice"]
        )
        - float(baseline_summary["segmentation"]["segmentation_dice"]),
        "selection_score": float(candidate_summary["composite"]["selection_score"])
        - float(baseline_summary["composite"]["selection_score"]),
    }
    if not all(math.isfinite(value) for value in gains.values()):
        raise ValueError("non-finite matched-fold gain")
    gates = {
        "overall_classification_gain": gains["average_precision"]
        >= minimum_classification_gain,
        "weak_source_classification_gain": gains[
            "weak_sources_average_precision"
        ]
        >= minimum_weak_source_gain,
        "lung_cell_classification_gain": gains["lung_cells_average_precision"]
        >= minimum_lung_cell_gain,
        "mature_classification_preserved": gains["mature_average_precision"]
        >= -maximum_mature_source_drop,
        "segmentation_preserved": gains["segmentation_dice"]
        >= -maximum_segmentation_drop,
        "composite_preserved": gains["selection_score"]
        >= minimum_composite_gain,
    }
    decision = (
        "EXPAND_PROMPT_MASKED_CLASSIFIER"
        if all(gates.values())
        else "KEEP_GLOBAL_CLASSIFIER"
    )
    return decision, gains, gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--candidate-history", type=Path, required=True)
    parser.add_argument("--baseline-source-audit", type=Path, required=True)
    parser.add_argument("--candidate-source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-classification-gain", type=float, default=0.005)
    parser.add_argument("--minimum-weak-source-gain", type=float, default=0.010)
    parser.add_argument("--minimum-lung-cell-gain", type=float, default=0.005)
    parser.add_argument("--maximum-mature-source-drop", type=float, default=0.005)
    parser.add_argument("--maximum-segmentation-drop", type=float, default=0.005)
    parser.add_argument("--minimum-composite-gain", type=float, default=0.0)
    args = parser.parse_args()
    thresholds = {
        "minimum_classification_gain": args.minimum_classification_gain,
        "minimum_weak_source_gain": args.minimum_weak_source_gain,
        "minimum_lung_cell_gain": args.minimum_lung_cell_gain,
        "maximum_mature_source_drop": args.maximum_mature_source_drop,
        "maximum_segmentation_drop": args.maximum_segmentation_drop,
        "minimum_composite_gain": args.minimum_composite_gain,
    }
    if any(value < 0 for value in thresholds.values()):
        parser.error("gate thresholds must be nonnegative")

    baseline_history = json.loads(args.baseline_history.read_text(encoding="utf-8"))
    candidate_history = json.loads(args.candidate_history.read_text(encoding="utf-8"))
    if not isinstance(baseline_history, list) or not isinstance(candidate_history, list):
        raise ValueError("training histories must be JSON lists")
    baseline_audit = json.loads(
        args.baseline_source_audit.read_text(encoding="utf-8")
    )
    candidate_audit = json.loads(
        args.candidate_source_audit.read_text(encoding="utf-8")
    )
    baseline_summary = summarize(baseline_history)
    candidate_summary = summarize(candidate_history)
    decision, gains, gates = decide(
        baseline_summary,
        candidate_summary,
        baseline_audit,
        candidate_audit,
        **thresholds,
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "intervention": {
            "prompt_masked_classifier": True,
            "pooling": ["global_average", "global_max", "prompt_average", "prompt_max"],
        },
        "baseline_history": str(args.baseline_history.resolve()),
        "baseline_history_sha256": file_sha256(args.baseline_history),
        "candidate_history": str(args.candidate_history.resolve()),
        "candidate_history_sha256": file_sha256(args.candidate_history),
        "baseline_source_audit": str(args.baseline_source_audit.resolve()),
        "baseline_source_audit_sha256": file_sha256(args.baseline_source_audit),
        "candidate_source_audit": str(args.candidate_source_audit.resolve()),
        "candidate_source_audit_sha256": file_sha256(args.candidate_source_audit),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "gains": gains,
        "gates": gates,
        "thresholds": thresholds,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
