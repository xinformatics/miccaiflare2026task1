#!/usr/bin/env python3
"""Choose a checkpoint family using leakage-free grouped OOF evidence only."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


CHECKPOINT_FILES = {
    "composite": "checkpoint_best.pt",
    "segmentation": "checkpoint_best_segmentation.pt",
    "cohort_segmentation": "checkpoint_best_cohort_segmentation.pt",
    "classification": "checkpoint_best_classification.pt",
}
PREDICTION_FILES = {
    "composite": "validation_predictions_best.npz",
    "segmentation": "validation_predictions_best_segmentation.npz",
    "cohort_segmentation": "validation_predictions_best_cohort_segmentation.npz",
    "classification": "validation_predictions_best_classification.npz",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"candidate must be NAME=SUMMARY: {value}")
    name, raw_path = value.split("=", 1)
    if name not in CHECKPOINT_FILES:
        raise ValueError(f"unsupported strategy: {name}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return name, path


def eligibility(summary: dict) -> bool:
    classification = summary["classification"]
    return bool(
        float(classification["average_precision"])
        >= float(classification["prevalence"]) + 0.15
        and float(classification["precision"]) >= 0.50
        and float(classification["recall"]) >= 0.20
    )


def choose(candidates: dict[str, dict]) -> str:
    legacy = {"composite", "segmentation", "classification"}
    extended = set(CHECKPOINT_FILES)
    if set(candidates) not in (legacy, extended):
        raise ValueError(
            "either the historical composite/segmentation/classification set "
            "or that complete set plus cohort_segmentation is required"
        )
    # Candidate rejection must be adequate before contour quality is compared.
    # Among adequate classifiers, segmentation Dice is the primary deployment
    # objective; AP breaks exact ties. If none passes, retain the same ordering
    # but record that public promotion is not justified by OOF evidence.
    return max(
        candidates,
        key=lambda name: (
            eligibility(candidates[name]),
            float(candidates[name]["segmentation"]["dice"]),
            float(candidates[name]["classification"]["average_precision"]),
            name == "composite",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-summary", type=Path, required=True)
    args = parser.parse_args()
    parsed = [parse_candidate(value) for value in args.candidate]
    if len(parsed) != len({name for name, _ in parsed}):
        parser.error("duplicate strategy")
    summaries = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in parsed
    }
    selected = choose(summaries)
    report = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "selection_basis": "grouped five-fold OOF only; no public labels",
        "selected_strategy": selected,
        "checkpoint_filename": CHECKPOINT_FILES[selected],
        "prediction_filename": PREDICTION_FILES[selected],
        "classification_eligible": eligibility(summaries[selected]),
        "strategies": {
            name: {
                "summary": str(path.resolve()),
                "summary_sha256": file_sha256(path),
                "classification_eligible": eligibility(summaries[name]),
                "average_precision": float(
                    summaries[name]["classification"]["average_precision"]
                ),
                "precision": float(summaries[name]["classification"]["precision"]),
                "recall": float(summaries[name]["classification"]["recall"]),
                "segmentation_dice": float(summaries[name]["segmentation"]["dice"]),
            }
            for name, path in parsed
        },
    }
    atomic_json(args.output, report)
    selected_summary = dict(summaries[selected])
    selected_summary["checkpoint_selection"] = report
    atomic_json(args.selected_summary, selected_summary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
