#!/usr/bin/env python3
"""Write a non-interrupting early-learning audit for the active ResEnc S6 run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from pathlib import Path


EXPECTED_HISTORICAL_SHA256 = (
    "e4f1a79b388b109ba9ce5473cfe1c92a12129a2b2223944ca27823dcec48121b"
)
PSEUDO_DICE_PATTERN = re.compile(
    r"Pseudo dice \[np\.(?:float32|float64)\(([-+0-9.eE]+)\)\]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_training_log(text: str) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.search(r"(?:^|: )Epoch (\d+)$", line)
        if match:
            current = {"epoch": int(match.group(1))}
            continue
        if current is None:
            continue
        fields = (
            ("learning_rate", r"(?:^|: )Current learning rate: ([-+0-9.eE]+)$"),
            ("train_loss", r"(?:^|: )train_loss ([-+0-9.eE]+)$"),
            ("val_loss", r"(?:^|: )val_loss ([-+0-9.eE]+)$"),
            ("epoch_seconds", r"(?:^|: )Epoch time: ([-+0-9.eE]+) s$"),
        )
        for name, pattern in fields:
            match = re.search(pattern, line)
            if match:
                current[name] = float(match.group(1))
        match = PSEUDO_DICE_PATTERN.search(line)
        if match:
            current["pseudo_dice"] = float(match.group(1))
        required = {
            "epoch",
            "learning_rate",
            "train_loss",
            "val_loss",
            "pseudo_dice",
            "epoch_seconds",
        }
        if required.issubset(current):
            records.append(dict(current))
            current = None
    return records


def audit(
    log_path: Path,
    historical_path: Path,
    minimum_completed_epochs: int = 10,
) -> dict[str, object]:
    if minimum_completed_epochs < 2:
        raise ValueError("minimum_completed_epochs must be at least two")
    if sha256(historical_path) != EXPECTED_HISTORICAL_SHA256:
        raise ValueError("historical ResEnc baseline identity changed")
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    trajectory = historical.get("trajectory", [])
    activation_epochs = [
        int(row["epoch"])
        for row in trajectory
        if float(row.get("lesion_dsc", 0.0)) > 0
    ]
    if not activation_epochs or min(activation_epochs) != 6:
        raise ValueError("historical foreground-activation reference changed")

    records = parse_training_log(log_path.read_text(encoding="utf-8"))
    completed = records[:minimum_completed_epochs]
    expected_epochs = list(range(minimum_completed_epochs))
    observed_epochs = [int(row["epoch"]) for row in completed]
    complete_prefix = observed_epochs == expected_epochs
    finite = complete_prefix and all(
        math.isfinite(float(row[key]))
        for row in completed
        for key in (
            "learning_rate",
            "train_loss",
            "val_loss",
            "pseudo_dice",
            "epoch_seconds",
        )
    )
    positive_epoch_times = finite and all(
        float(row["epoch_seconds"]) > 0 for row in completed
    )
    train_improved = finite and (
        float(completed[-1]["train_loss"]) < float(completed[0]["train_loss"])
    )
    validation_improved = finite and (
        float(completed[-1]["val_loss"]) < float(completed[0]["val_loss"])
    )
    activation = [
        int(row["epoch"])
        for row in completed
        if float(row["pseudo_dice"]) > 0
    ]

    if not (complete_prefix and finite and positive_epoch_times):
        decision = "REVIEW_REQUIRED_INCOMPLETE_OR_NONFINITE_TRAJECTORY"
    elif not (train_improved and validation_improved):
        decision = "REVIEW_REQUIRED_LOSS_TRAJECTORY"
    elif not activation:
        decision = "REVIEW_REQUIRED_NO_FOREGROUND_ACTIVATION"
    else:
        decision = "PASS_EARLY_LEARNING_SIGNAL"

    return {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": decision,
        "role": "non_interrupting_epoch10_sentinel",
        "log": {"path": str(log_path.resolve()), "sha256": sha256(log_path)},
        "historical_reference": {
            "path": str(historical_path.resolve()),
            "sha256": EXPECTED_HISTORICAL_SHA256,
            "first_nonzero_public_dsc_epoch": min(activation_epochs),
            "public_dsc_zero_through_epoch": min(activation_epochs) - 1,
        },
        "minimum_completed_epochs": minimum_completed_epochs,
        "observed_completed_epochs": observed_epochs,
        "checks": {
            "complete_contiguous_prefix": complete_prefix,
            "all_values_finite": finite,
            "positive_epoch_times": positive_epoch_times,
            "train_loss_improved": train_improved,
            "validation_loss_improved": validation_improved,
            "foreground_activated": bool(activation),
        },
        "first_foreground_activation_epoch": activation[0] if activation else None,
        "trajectory": completed,
        "automatic_training_stop": False,
        "training_mutated": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-completed-epochs", type=int, default=10)
    args = parser.parse_args()
    payload = audit(
        args.log,
        args.historical,
        minimum_completed_epochs=args.minimum_completed_epochs,
    )
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
