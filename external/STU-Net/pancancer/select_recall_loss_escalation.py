#!/usr/bin/env python3
"""Predeclared decision for escalating Tversky from 0.25/0.75 to 0.20/0.80."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


EPOCHS = (40, 80)
P70_DSC_GAIN_MINIMUM = 0.002
P70_NSD_GAIN_FLOOR = -0.003
MICRO_RECALL_GAIN_MINIMUM = 0.005
MICRO_PRECISION_LOSS_MAXIMUM = 0.03
PRECISION_RECALL_GAP_MINIMUM = 0.03
VOLUME_RATIO_MAXIMUM = 0.95
FN_ORACLE_ADVANTAGE_MINIMUM = 0.02


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def p70_metric(payload: dict) -> dict[str, float]:
    if "metrics" in payload:
        metric = payload["metrics"]["gf_anchor_confirm_either_p70"]
    else:
        matches = [
            row
            for row in payload.get("results", [])
            if row.get("configuration") == "gf_anchor_confirm_either_p70"
        ]
        if len(matches) != 1:
            raise ValueError("exactly one fixed-p70 result is required")
        metric = matches[0]
    return {
        "Lesion_DSC": float(metric["Lesion_DSC"]),
        "Lesion_NSD": float(metric["Lesion_NSD"]),
    }


def validate_endpoint(payload: dict, epoch: int) -> None:
    if (
        payload.get("status") != "COMPLETE"
        or int(payload.get("evaluation_epoch", -1)) != epoch
        or payload.get("hidden_read")
        or payload.get("submission_created")
        or payload.get("uploaded")
    ):
        raise ValueError(f"invalid epoch-{epoch} fixed-public endpoint")


def evaluate_epoch(
    epoch: int,
    balanced_terminal: Path,
    source_terminal: Path,
    balanced_lesion: Path,
    source_lesion: Path,
    balanced_gap: Path,
    source_gap: Path,
) -> dict:
    balanced_endpoint = load_json(balanced_terminal)
    source_endpoint = load_json(source_terminal)
    validate_endpoint(balanced_endpoint, epoch)
    validate_endpoint(source_endpoint, epoch)
    balanced_p70 = p70_metric(balanced_endpoint)
    source_p70 = p70_metric(source_endpoint)
    balanced_errors = load_json(balanced_lesion)
    source_errors = load_json(source_lesion)
    balanced_volume = load_json(balanced_gap)
    source_volume = load_json(source_gap)
    if balanced_errors.get("cases") != 50 or source_errors.get("cases") != 50:
        raise ValueError("exactly 50 public lesion-error cases are required")
    if balanced_volume.get("cases") != 50 or source_volume.get("cases") != 50:
        raise ValueError("exactly 50 public volume-bias cases are required")

    balanced_micro = balanced_errors["micro"]
    source_micro = source_errors["micro"]
    source_macro = source_errors["macro"]
    p70_dsc_gain = source_p70["Lesion_DSC"] - balanced_p70["Lesion_DSC"]
    p70_nsd_gain = source_p70["Lesion_NSD"] - balanced_p70["Lesion_NSD"]
    recall_gain = float(source_micro["recall"]) - float(balanced_micro["recall"])
    precision_loss = float(balanced_micro["precision"]) - float(source_micro["precision"])
    precision_recall_gap = float(source_micro["precision"]) - float(source_micro["recall"])
    volume_ratio = float(source_volume["aggregate"]["volume_ratio"])
    fn_oracle_advantage = float(source_macro["fn_free_oracle_dice"]) - float(
        source_macro["fp_free_oracle_dice"]
    )
    checks = {
        "p70_dsc_direction": p70_dsc_gain >= P70_DSC_GAIN_MINIMUM,
        "p70_nsd_preserved": p70_nsd_gain >= P70_NSD_GAIN_FLOOR,
        "micro_recall_responded": recall_gain >= MICRO_RECALL_GAIN_MINIMUM,
        "micro_precision_preserved": precision_loss <= MICRO_PRECISION_LOSS_MAXIMUM,
        "remaining_undersegmentation": (
            precision_recall_gap >= PRECISION_RECALL_GAP_MINIMUM
            or volume_ratio <= VOLUME_RATIO_MAXIMUM
        ),
        "fn_headroom_dominates": fn_oracle_advantage >= FN_ORACLE_ADVANTAGE_MINIMUM,
    }
    return {
        "epoch": epoch,
        "balanced_p70": balanced_p70,
        "source_p70": source_p70,
        "p70_dsc_gain": p70_dsc_gain,
        "p70_nsd_gain": p70_nsd_gain,
        "balanced_micro_precision": float(balanced_micro["precision"]),
        "balanced_micro_recall": float(balanced_micro["recall"]),
        "source_micro_precision": float(source_micro["precision"]),
        "source_micro_recall": float(source_micro["recall"]),
        "micro_recall_gain": recall_gain,
        "micro_precision_loss": precision_loss,
        "source_precision_minus_recall": precision_recall_gap,
        "source_aggregate_volume_ratio": volume_ratio,
        "source_fn_oracle_advantage": fn_oracle_advantage,
        "checks": checks,
        "qualifies": all(checks.values()),
        "assets": {
            "balanced_terminal": asset(balanced_terminal),
            "source_terminal": asset(source_terminal),
            "balanced_lesion_errors": asset(balanced_lesion),
            "source_lesion_errors": asset(source_lesion),
            "balanced_volume_bias": asset(balanced_gap),
            "source_volume_bias": asset(source_gap),
        },
    }


def select(balanced_root: Path, source_root: Path, diagnostic_root: Path) -> dict:
    rows = []
    for epoch in EPOCHS:
        tag = f"epoch_{epoch:04d}"
        rows.append(
            evaluate_epoch(
                epoch,
                balanced_root / tag / "TERMINAL.json",
                source_root / tag / "TERMINAL.json",
                diagnostic_root / f"balanced_{tag}" / "lesion_errors" / "summary.json",
                diagnostic_root / f"source_{tag}" / "lesion_errors" / "summary.json",
                diagnostic_root / f"balanced_{tag}" / "dsc_nsd_gap" / "summary.json",
                diagnostic_root / f"source_{tag}" / "dsc_nsd_gap" / "summary.json",
            )
        )
    qualified = [row for row in rows if row["qualifies"]]
    selected = max(
        qualified or rows,
        key=lambda row: (
            sum(row["checks"].values()),
            row["p70_dsc_gain"],
            row["p70_nsd_gain"],
        ),
    )
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": (
            "QUALIFY_TVERSKY_A020_B080_FOLLOWUP"
            if qualified
            else "KEEP_A025_B075_NO_STRONGER_LOSS"
        ),
        "selection_basis": (
            "same-epoch balanced-vs-a025b075 fixed-public causal comparison; "
            "a020b080 only when recall responds, precision/NSD are preserved, "
            "and FN/volume evidence shows residual undersegmentation"
        ),
        "thresholds": {
            "p70_dsc_gain_minimum": P70_DSC_GAIN_MINIMUM,
            "p70_nsd_gain_floor": P70_NSD_GAIN_FLOOR,
            "micro_recall_gain_minimum": MICRO_RECALL_GAIN_MINIMUM,
            "micro_precision_loss_maximum": MICRO_PRECISION_LOSS_MAXIMUM,
            "precision_recall_gap_minimum": PRECISION_RECALL_GAP_MINIMUM,
            "volume_ratio_maximum": VOLUME_RATIO_MAXIMUM,
            "fn_oracle_advantage_minimum": FN_ORACLE_ADVANTAGE_MINIMUM,
        },
        "epochs": rows,
        "selected": selected,
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = select(args.balanced_root, args.source_root, args.diagnostic_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
