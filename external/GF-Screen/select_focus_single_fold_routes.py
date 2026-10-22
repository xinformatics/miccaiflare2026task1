#!/usr/bin/env python3
"""Select per-source Focus epochs from one held-out-fold diagnostic trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


GROUPS = ("mature", "weak_sources", "lung_cells")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_routes(audits: list[Path], expected_epochs: int) -> dict[str, object]:
    if expected_epochs < 2 or len(audits) != expected_epochs:
        raise ValueError("single-fold route selection requires every expected epoch")
    rows = []
    for path in audits:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("public_read")
            or payload.get("hidden_read")
            or payload.get("submission_created")
            or set(payload.get("groups", {})) != set(GROUPS)
        ):
            raise ValueError(f"invalid local-only Focus audit: {path}")
        rows.append((path, payload))
    folds = {int(payload["fold"]) for _, payload in rows}
    seeds = {int(payload["seed"]) for _, payload in rows}
    index_hashes = {str(payload["index_sha256"]) for _, payload in rows}
    epochs = {int(payload["epoch"]) for _, payload in rows}
    if (
        len(folds) != 1
        or len(seeds) != 1
        or len(index_hashes) != 1
        or epochs != set(range(expected_epochs))
    ):
        raise ValueError("Focus audit trajectory identity/epoch mismatch")
    selected = {}
    for group in GROUPS:
        path, payload = max(
            rows,
            key=lambda item: (
                float(item[1]["groups"][group]["selection_score"]),
                float(item[1]["groups"][group]["case_balanced_dice"]),
                float(item[1]["groups"][group]["cohort_balanced_dice"]),
                -int(item[1]["epoch"]),
            ),
        )
        selected[group] = {
            "epoch": int(payload["epoch"]),
            "checkpoint": payload["checkpoint"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "prediction": payload["prediction"],
            "prediction_sha256": payload["prediction_sha256"],
            "audit": str(path.resolve()),
            "audit_sha256": sha256(path),
            "metrics": payload["groups"][group],
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "one case-grouped held-out fold; TTA pilot routing only",
        "fold": next(iter(folds)),
        "seed": next(iter(seeds)),
        "index_sha256": next(iter(index_hashes)),
        "expected_epochs": expected_epochs,
        "selected": selected,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = select_routes(args.audit, args.expected_epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
