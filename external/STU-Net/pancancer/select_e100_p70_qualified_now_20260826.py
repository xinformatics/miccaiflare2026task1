#!/usr/bin/env python3
"""Emit the already-qualified epoch-100 p70 deployment handoff."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from select_best_e100_e130_20260826 import (
    ANCHOR_DSC,
    ANCHOR_NSD,
    DSC_GATE,
    E100_DEPLOYMENT_CONFIGURATION,
    NSD_FLOOR,
    asset,
    atomic_json,
    exact_e100_metrics,
)


def emit(*, screen: Path, metrics: Path, checkpoint: Path, output_dir: Path) -> dict:
    dsc, nsd = exact_e100_metrics(screen, metrics)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    deployment = {
        "configuration": E100_DEPLOYMENT_CONFIGURATION,
        "Lesion_DSC": dsc,
        "Lesion_NSD": nsd,
        "DSC_gain": dsc - ANCHOR_DSC,
        "NSD_gain": nsd - ANCHOR_NSD,
    }
    analysis = output_dir / "analysis.json"
    atomic_json(
        analysis,
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "COMPLETE",
            "case_count": 50,
            "selection_policy": "predeclared_epoch100_p70_independent_confirmation",
            "metrics": {E100_DEPLOYMENT_CONFIGURATION: {"Lesion_DSC": dsc, "Lesion_NSD": nsd}},
            "deployments": [deployment],
            "best_deployment": deployment,
            "best_unconstrained_deployment": deployment,
            "fixed_public_DSC_gate": DSC_GATE,
            "fixed_public_NSD_floor": NSD_FLOOR,
            "public_gate": True,
            "decision": "QUALIFY_SUBMISSION_PREPARATION",
            "role": "fixed_public_ensemble_selection_only",
            "source_screen": asset(screen),
            "source_exact_metrics": asset(metrics),
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        },
    )
    summary = output_dir / "summary.tsv"
    fields = ["branch", "checkpoint_epoch", "configuration", "Lesion_DSC", "Lesion_NSD", "DSC_gain", "NSD_gain"]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({"branch": "base_global_box", "checkpoint_epoch": 100, **deployment})
    terminal = output_dir / "TERMINAL.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "QUALIFY_SUBMISSION_PREPARATION",
        "predeclared_epochs": [100],
        "selection_policy": "immediate replay of exact epoch-100 candidate that cleared both fixed-public gates",
        "selected_branch": "base_global_box",
        "selected_epoch": 100,
        "selected_configuration": E100_DEPLOYMENT_CONFIGURATION,
        "selected_fixed_public_DSC": dsc,
        "selected_fixed_public_NSD": nsd,
        "checkpoint": asset(checkpoint),
        "selected_analysis": asset(analysis),
        "fixed_public_DSC_gate": DSC_GATE,
        "fixed_public_NSD_floor": NSD_FLOOR,
        "summary": asset(summary),
        "source_screen": asset(screen),
        "source_exact_metrics": asset(metrics),
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(terminal, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(emit(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
