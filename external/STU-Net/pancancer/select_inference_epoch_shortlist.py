#!/usr/bin/env python3
"""Select spaced every-epoch STU-Net snapshots for public inference.

Selection uses only the trainer's trusted-box internal diagnostics. It is a
compute-allocation heuristic, never a promotion decision; fixed-public DSC and
NSD remain mandatory before packaging a hidden candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


METRIC = "ema_fg_dice"


def read_inventory(fold: Path, start_epoch: int, end_epoch: int) -> list[dict]:
    rows = []
    for sidecar in sorted(fold.glob("checkpoint_inference_epoch_*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"unsupported sidecar schema: {sidecar}")
        epoch = int(payload["current_epoch"])
        if not start_epoch <= epoch <= end_epoch:
            continue
        checkpoint = fold / str(payload["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_bytes = checkpoint.stat().st_size
        if int(payload["checkpoint_bytes"]) != actual_bytes:
            raise ValueError(f"checkpoint size mismatch: {checkpoint}")
        metrics = payload.get("internal_partial_label_metrics", {})
        score = metrics.get(METRIC)
        if score is None or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"invalid {METRIC}: {sidecar}")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_bytes": actual_bytes,
                "trainer_name": payload["trainer_name"],
                "internal_partial_label_metrics": metrics,
            }
        )
    epochs = [row["epoch"] for row in rows]
    if len(epochs) != len(set(epochs)):
        raise ValueError(f"duplicate epochs in {fold}")
    return rows


def select_spaced(rows: list[dict], top_k: int, min_separation: int) -> list[dict]:
    if not rows:
        return []
    by_epoch = {row["epoch"]: row for row in rows}
    endpoint = by_epoch[max(by_epoch)]
    selected = [endpoint]
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["internal_partial_label_metrics"][METRIC]),
            -row["epoch"],
        ),
    )
    for row in ranked:
        if row in selected:
            continue
        if all(abs(row["epoch"] - item["epoch"]) >= min_separation for item in selected):
            selected.append(row)
        if len(selected) == top_k:
            break
    return sorted(selected, key=lambda row: row["epoch"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=Path, action="append", required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--end-epoch", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-separation", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_epoch < 0 or args.end_epoch < args.start_epoch:
        parser.error("invalid epoch interval")
    if args.top_k < 1 or args.min_separation < 1:
        parser.error("top-k and min-separation must be positive")

    branches = []
    for fold in args.fold:
        rows = read_inventory(fold, args.start_epoch, args.end_epoch)
        if not rows:
            raise ValueError(f"no inference checkpoints in requested interval: {fold}")
        selected = select_spaced(rows, args.top_k, args.min_separation)
        branches.append(
            {
                "fold": str(fold.resolve()),
                "available_epochs": [row["epoch"] for row in rows],
                "selected": selected,
            }
        )
    output = {
        "schema_version": 1,
        "selection_role": "public-inference compute allocation only",
        "selection_metric": f"internal_partial_label_metrics.{METRIC}",
        "promotion_rule": "fixed-public DSC >= 0.6154316379500749 and NSD >= 0.5104225107769019",
        "start_epoch": args.start_epoch,
        "end_epoch": args.end_epoch,
        "top_k": args.top_k,
        "min_epoch_separation": args.min_separation,
        "branches": branches,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
