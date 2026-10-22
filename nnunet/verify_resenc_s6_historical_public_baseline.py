#!/usr/bin/env python3
"""Verify the diagnostic legacy ResEnc trajectory against its fixed-public CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_metrics(path: Path) -> dict[str, int | float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["Name"] for row in rows]
    if len(rows) != 50 or len(set(names)) != 50:
        raise ValueError(f"{path}: expected 50 unique cases, found {len(rows)}/{len(set(names))}")
    dsc = [float(row["Lesion_DSC"]) for row in rows]
    nsd = [float(row["Lesion_NSD"]) for row in rows]
    return {
        "lesion_dsc": sum(dsc) / len(dsc),
        "lesion_nsd": sum(nsd) / len(nsd),
        "zero_dsc_cases": sum(value == 0.0 for value in dsc),
    }


def verify(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if manifest["role"] != "diagnostic_only_fixed_public_baseline":
        raise ValueError("manifest is not explicitly diagnostic-only")
    if manifest["dataset_role"] != "fixed_public_validation_only":
        raise ValueError("manifest has an unexpected dataset role")
    if manifest["submission_created"] or manifest["hidden_data_read"]:
        raise ValueError("diagnostic baseline cannot use hidden data or create a submission")

    metrics_root = Path(manifest["metrics_root"])
    trajectory = manifest["trajectory"]
    if [row["epoch"] for row in trajectory] != list(range(26)):
        raise ValueError("expected the complete, ordered epoch-0-through-25 trajectory")

    for expected in trajectory:
        epoch = expected["epoch"]
        csv_path = metrics_root / f"epoch_{epoch:04d}" / "challenge_metrics.csv"
        observed = aggregate_metrics(csv_path)
        for metric in ("lesion_dsc", "lesion_nsd"):
            if not math.isclose(observed[metric], expected[metric], rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"epoch {epoch}: {metric} mismatch: "
                    f"manifest={expected[metric]} csv={observed[metric]}"
                )
        if observed["zero_dsc_cases"] != expected["zero_dsc_cases"]:
            raise ValueError(f"epoch {epoch}: zero-DSC count mismatch")

    best = max(trajectory, key=lambda row: row["lesion_dsc"])
    declared_best = manifest["best_observed"]
    for key in ("epoch", "lesion_dsc", "lesion_nsd", "zero_dsc_cases"):
        if best[key] != declared_best[key]:
            raise ValueError(f"declared best does not match trajectory for {key}")

    terminal = manifest["terminal_observed"]
    if terminal["epoch"] != trajectory[-1]["epoch"]:
        raise ValueError("terminal epoch does not match the trajectory")
    for record in (declared_best, terminal):
        csv_path = metrics_root / f"epoch_{record['epoch']:04d}" / "challenge_metrics.csv"
        if sha256(csv_path) != record["metrics_sha256"]:
            raise ValueError(f"epoch {record['epoch']}: CSV SHA-256 mismatch")

    gate = manifest["corrected_resenc_comparison"]
    if gate["promotion_gate_dsc"] <= declared_best["lesion_dsc"]:
        raise ValueError("promotion gate was weakened to the diagnostic baseline")
    if gate["promotion_gate_nsd"] <= declared_best["lesion_nsd"]:
        raise ValueError("promotion NSD gate was weakened to the diagnostic baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("RESENC_S6_HISTORICAL_PUBLIC_BASELINE_20260827.json"),
    )
    args = parser.parse_args()
    verify(args.manifest.resolve())
    print("PASS: historical ResEnc public baseline matches all 26 fixed-public CSVs")


if __name__ == "__main__":
    main()
