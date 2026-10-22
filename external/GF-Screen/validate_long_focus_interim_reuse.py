#!/usr/bin/env python3
"""Validate an exact public-only long-Focus endpoint for trajectory reuse."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VARIANTS = {"deployed500", "deployed1000", "touching"}
DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_identity(identity: object, expected: Path, name: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"missing {name} identity")
    if (
        Path(str(identity.get("path", ""))).resolve() != expected.resolve()
        or identity.get("sha256") != sha256(expected)
    ):
        raise ValueError(f"{name} identity mismatch")


def require_within(path: Path, root: Path, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes interim root") from error
    return resolved


def validate_interim(source: Path, checkpoint: Path, epoch: int) -> dict:
    if epoch not in {293, 303}:
        raise ValueError("only predeclared epochs 293 and 303 may be reused")
    source = source.resolve()
    terminal_path = source / "TERMINAL.json"
    summary_path = source / "summary.tsv"
    if not checkpoint.is_file() or not terminal_path.is_file() or not summary_path.is_file():
        raise ValueError("interim reuse asset is missing")

    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "COMPLETE"
        or terminal.get("decision") != "KEEP_TOUCHCOMP"
        or terminal.get("selected_variant") is not None
        or float(terminal.get("glance_threshold", -1)) != 0.20
        or float(terminal.get("fixed_public_DSC_gate", -1)) != DSC_GATE
        or float(terminal.get("fixed_public_NSD_floor", -1)) != NSD_FLOOR
        or terminal.get("public_read") is not True
        or terminal.get("hidden_read") is not False
        or terminal.get("submission_created") is not False
        or terminal.get("uploaded") is not False
    ):
        raise ValueError("interim terminal contract mismatch")
    require_identity(terminal.get("checkpoint"), checkpoint, "checkpoint")
    require_identity(terminal.get("summary"), summary_path, "summary")

    with summary_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 3 or {row.get("variant") for row in rows} != VARIANTS:
        raise ValueError("interim deployment coverage mismatch")
    for row in rows:
        if row.get("gate") != "KEEP_ANCHOR":
            raise ValueError("a reusable rejected endpoint must not contain PROMOTE")
        predictions = require_within(Path(row["predictions"]), source, "predictions")
        metrics = require_within(Path(row["metrics"]), source, "metrics")
        robust = require_within(Path(row["robust"]), source, "robust gate")
        prediction_files = sorted(predictions.glob("*.nii.gz"))
        if len(prediction_files) != 50 or any(path.stat().st_size < 1 for path in prediction_files):
            raise ValueError("interim prediction coverage mismatch")
        with metrics.open(encoding="utf-8") as handle:
            metric_rows = list(csv.DictReader(handle))
        if len(metric_rows) != 50 or len({row["Name"] for row in metric_rows}) != 50:
            raise ValueError("interim metric coverage mismatch")
        robust_payload = json.loads(robust.read_text(encoding="utf-8"))
        if robust_payload.get("decision") != "KEEP_ANCHOR":
            raise ValueError("interim robust decision mismatch")
    return {
        "status": "PASS",
        "epoch": epoch,
        "source": str(source),
        "checkpoint_sha256": sha256(checkpoint),
        "summary_sha256": sha256(summary_path),
        "variants": sorted(VARIANTS),
        "cases_per_variant": 50,
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(validate_interim(args.source, args.checkpoint, args.epoch), indent=2))


if __name__ == "__main__":
    main()
