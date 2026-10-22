#!/usr/bin/env python3
"""Combine label-free stage replays into one per-case deployment budget.

This does not run inference or read validation labels.  It joins the measured
BiomedParse, shared-Focus GF, and proxy-sparse STU timings by case and reports
both the steady-state (models initialized once) and conservative cold-start
budgets.  It deliberately does not authorize the final Docker by itself.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def canonical_case(name: str) -> str:
    for suffix in (".nii.gz", ".npz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prediction_payload(equivalence: dict, equivalence_path: Path) -> tuple[dict, Path]:
    reference = equivalence.get("full_prediction") or {}
    prediction_path = Path(reference.get("path", ""))
    if not prediction_path.is_file():
        raise FileNotFoundError(f"missing full prediction terminal from {equivalence_path}")
    expected_sha = reference.get("sha256")
    if not expected_sha or sha256(prediction_path) != expected_sha:
        raise RuntimeError(f"full prediction terminal hash mismatch: {prediction_path}")
    return read_json(prediction_path), prediction_path


def indexed(rows: list[dict], seconds_key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        case = canonical_case(str(row["case"]))
        if case in result:
            raise RuntimeError(f"duplicate case {case}")
        if float(row[seconds_key]) < 0:
            raise RuntimeError(f"negative runtime for {case}")
        result[case] = row
    return result


def summarize(args: argparse.Namespace) -> dict:
    paths = {
        "biomedparse": args.biomedparse,
        "gf": args.gf,
        "stunet": args.stunet,
    }
    equivalence = {name: read_json(path) for name, path in paths.items()}
    required = {
        "biomedparse": "exact_prompt_outputs",
        "gf": "exact_raw_route_equivalence",
        "stunet": "exact_full_volume_proxy_equivalence",
    }
    problems = []
    for name, flag in required.items():
        payload = equivalence[name]
        if payload.get("status") != "COMPLETE":
            problems.append(f"{name}:status={payload.get('status')}")
        if not payload.get(flag):
            problems.append(f"{name}:{flag}=false")
        if any(
            payload.get(field)
            for field in (
                "public_labels_read",
                "hidden_read",
                "submission_created",
                "uploaded",
            )
        ):
            raise RuntimeError(f"forbidden data/action flag in {paths[name]}")
    common = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "deployment_authorized": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "public_images_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    if problems:
        return {
            **common,
            "status": "COMPLETE",
            "decision": "REJECT_INCOMPLETE_RUNTIME_CHAIN",
            "problems": problems,
            "case_count": 0,
            "cases": [],
        }

    biomed_prediction, biomed_path = prediction_payload(
        equivalence["biomedparse"], paths["biomedparse"]
    )
    gf_prediction, gf_path = prediction_payload(equivalence["gf"], paths["gf"])
    biomed_rows = indexed(biomed_prediction["cases"], "wall_seconds")
    gf_rows = indexed(gf_prediction["cases"], "wall_seconds")
    stunet_rows = indexed(equivalence["stunet"]["cases"], "stage_seconds")
    names = sorted(set(biomed_rows) | set(gf_rows) | set(stunet_rows))
    if len(names) != 50 or any(set(rows) != set(names) for rows in (biomed_rows, gf_rows, stunet_rows)):
        raise RuntimeError("runtime stage inventories do not match exactly at 50 cases")

    rows = []
    for name in names:
        biomed_seconds = float(biomed_rows[name]["wall_seconds"])
        gf_seconds = float(gf_rows[name]["wall_seconds"])
        stunet_seconds = float(stunet_rows[name]["stage_seconds"])
        rows.append(
            {
                "case": name,
                "biomedparse_seconds": biomed_seconds,
                "gf_seconds": gf_seconds,
                "stunet_seconds": stunet_seconds,
                "steady_state_seconds": biomed_seconds + gf_seconds + stunet_seconds,
                "stunet_skipped": bool(stunet_rows[name]["stunet_skipped"]),
            }
        )
    times = [row["steady_state_seconds"] for row in rows]
    initialization = {
        "biomedparse": float(biomed_prediction["initialization_seconds"]),
        "gf": float(gf_prediction["model_initialization_seconds"]),
        "stunet": float(equivalence["stunet"].get("model_initialization_seconds", 0.0)),
    }
    initialization_total = sum(initialization.values())
    peak_reserved = {
        name: float(equivalence[name]["peak_gpu_memory_mb"]["reserved"])
        for name in paths
    }
    steady = {
        "mean": sum(times) / len(times),
        "median": quantile(times, 0.50),
        "p90": quantile(times, 0.90),
        "p95": quantile(times, 0.95),
        "max": max(times),
    }
    steady_60 = steady["max"] <= args.hard_seconds
    steady_target = steady["max"] <= args.target_seconds
    return {
        **common,
        "status": "COMPLETE",
        "decision": (
            "QUALIFY_STEADY_STATE_RUNTIME"
            if steady_60
            else "REJECT_STEADY_STATE_RUNTIME"
        ),
        "case_count": len(rows),
        "stage_prediction_terminals": {
            "biomedparse": str(biomed_path.resolve()),
            "gf": str(gf_path.resolve()),
            "stunet": "embedded in the STU equivalence terminal",
        },
        "model_initialization_seconds": {
            **initialization,
            "total": initialization_total,
            "amortized_per_case_at_500": initialization_total / 500.0,
        },
        "runtime_seconds": {
            "steady_state": steady,
            "cold_start_conservative_max": steady["max"] + initialization_total,
            "hard_limit": args.hard_seconds,
            "operating_target": args.target_seconds,
        },
        "runtime_eligible_60s_steady_state": steady_60,
        "runtime_eligible_50s_steady_state": steady_target,
        "memory": {
            "stage_peak_reserved_mb": peak_reserved,
            "sequential_stage_lower_bound_mb": max(peak_reserved.values()),
            "all_stage_peak_sum_conservative_mb": sum(peak_reserved.values()),
            "hard_limit_mb": args.hard_memory_mb,
            "perfect_efficiency_target_mb": args.efficiency_memory_mb,
            "sequential_stage_memory_eligible_16gb": max(peak_reserved.values())
            <= args.hard_memory_mb,
            "sequential_stage_memory_eligible_4gb": max(peak_reserved.values())
            <= args.efficiency_memory_mb,
        },
        "interpretation": (
            "steady-state assumes one persistent process initializes each model once; "
            "the cold-start value is conservative and final Docker end-to-end timing remains required"
        ),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--biomedparse", required=True, type=Path)
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--stunet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hard-seconds", default=60.0, type=float)
    parser.add_argument("--target-seconds", default=50.0, type=float)
    parser.add_argument("--hard-memory-mb", default=16384.0, type=float)
    parser.add_argument("--efficiency-memory-mb", default=4096.0, type=float)
    args = parser.parse_args()
    if not 0 < args.target_seconds <= args.hard_seconds:
        raise ValueError("runtime target must be positive and no larger than hard limit")
    payload = summarize(args)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
