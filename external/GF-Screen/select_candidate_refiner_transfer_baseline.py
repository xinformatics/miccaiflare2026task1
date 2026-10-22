#!/usr/bin/env python3
"""Select the strongest fixed checkpoint family on a cross-index transfer fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


EXPECTED_FAMILIES = {"composite", "segmentation", "classification"}


def select(records: dict[str, dict], expected_fold: int = 0) -> dict:
    if set(records) != EXPECTED_FAMILIES:
        raise ValueError("transfer baseline requires all three checkpoint families")
    index_hashes = {str(record.get("evaluation_index_sha256", "")) for record in records.values()}
    folds = {int(record.get("fold", -1)) for record in records.values()}
    seeds = {int(record.get("seed", -1)) for record in records.values()}
    if (
        expected_fold < 0
        or len(index_hashes) != 1
        or "" in index_hashes
        or folds != {expected_fold}
        or len(seeds) != 1
    ):
        raise ValueError("transfer records do not share the requested exact-fold evaluation")
    checkpoint_paths = {str(record.get("checkpoint", "")) for record in records.values()}
    if len(checkpoint_paths) != 3 or "" in checkpoint_paths:
        raise ValueError("checkpoint families are missing or duplicated")
    for family, record in records.items():
        if record.get("cross_index_transfer") is not True:
            raise ValueError(f"{family} is not an explicit cross-index transfer")
        if record.get("public_read") is not False or record.get("hidden_read") is not False:
            raise ValueError(f"{family} read a forbidden evaluation domain")
        metrics = record.get("metrics", {})
        for name in ("selection_score", "case_balanced_segmentation_dice", "average_precision"):
            value = float(metrics[name])
            if not 0 <= value <= 1:
                raise ValueError(f"invalid {family} metric: {name}")
    selected = max(
        records,
        key=lambda family: (
            float(records[family]["metrics"]["selection_score"]),
            float(records[family]["metrics"]["case_balanced_segmentation_dice"]),
            float(records[family]["metrics"]["average_precision"]),
            family,
        ),
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "decision": "TRANSFER_BASELINE_FROZEN",
        "selected_family": selected,
        "selected_checkpoint": records[selected]["checkpoint"],
        "selected_metrics": records[selected]["metrics"],
        "evaluation_index_sha256": next(iter(index_hashes)),
        "fold": expected_fold,
        "seed": next(iter(seeds)),
        "records": records,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--expected-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: dict[str, dict] = {}
    for value in args.candidate:
        family, separator, raw_path = value.partition("=")
        if not separator or family in records:
            raise ValueError("candidate must be unique FAMILY=PATH")
        path = Path(raw_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["result_path"] = str(path.resolve())
        payload["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        records[family] = payload
    result = select(records, args.expected_fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
