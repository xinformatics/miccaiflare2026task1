#!/usr/bin/env python3
"""Select one public-qualified epoch-100/130 STU-Net deployment for replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019
ANCHOR_DSC = 0.6004316379500749
ANCHOR_NSD = 0.5104225107769019
E100_SCREEN_CONFIGURATION = "either_p70_any"
E100_DEPLOYMENT_CONFIGURATION = "gf_anchor_confirm_either_p70"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def exact_e100_metrics(screen_path: Path, metrics_path: Path) -> tuple[float, float]:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if (
        screen.get("status") != "COMPLETE"
        or screen.get("case_count") != 50
        or screen.get("hidden_read")
        or screen.get("submission_created")
        or screen.get("uploaded")
        or float(screen.get("anchor_DSC", -1)) != ANCHOR_DSC
    ):
        raise ValueError("invalid epoch-100 DSC screen")
    row = next(
        (
            item
            for item in screen.get("configurations", ())
            if item.get("configuration") == E100_SCREEN_CONFIGURATION
        ),
        None,
    )
    if row is None or not row.get("passes_dsc_gate"):
        raise ValueError("epoch-100 p70 screen did not pass DSC gate")

    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["Name"] for row in rows]
    if len(rows) != 50 or len(set(names)) != 50:
        raise ValueError("epoch-100 exact metric inventory must contain 50 cases")
    dsc = sum(float(item["Lesion_DSC"]) for item in rows) / len(rows)
    nsd = sum(float(item["Lesion_NSD"]) for item in rows) / len(rows)
    if abs(dsc - float(row["Lesion_DSC"])) > 1e-12:
        raise ValueError("epoch-100 DSC screen/exact mismatch")
    if dsc < DSC_GATE or nsd < NSD_FLOOR:
        raise ValueError("epoch-100 exact candidate missed public gates")
    return dsc, nsd


def select_best(
    *,
    e100_screen: Path,
    e100_metrics: Path,
    e100_checkpoint: Path,
    e130_terminal: Path,
    output_dir: Path,
) -> dict:
    dsc100, nsd100 = exact_e100_metrics(e100_screen, e100_metrics)
    if not e100_checkpoint.is_file():
        raise FileNotFoundError(e100_checkpoint)

    e130 = json.loads(e130_terminal.read_text(encoding="utf-8"))
    if (
        e130.get("status") != "COMPLETE"
        or e130.get("predeclared_epochs") != [130]
        or e130.get("hidden_read")
        or e130.get("submission_created")
        or e130.get("uploaded")
    ):
        raise ValueError("invalid epoch-130 public terminal")
    summary130 = Path(e130.get("summary", {}).get("path", ""))
    if (
        not summary130.is_file()
        or e130.get("summary", {}).get("sha256") != sha256(summary130)
    ):
        raise ValueError("epoch-130 summary identity mismatch")
    with summary130.open(newline="", encoding="utf-8") as handle:
        rows130 = list(csv.DictReader(handle, delimiter="\t"))
    if not rows130:
        raise ValueError("empty epoch-130 public summary")

    e100_analysis = output_dir / "e100_anchor_confirm_either_p70_analysis.json"
    deployment100 = {
        "configuration": E100_DEPLOYMENT_CONFIGURATION,
        "Lesion_DSC": dsc100,
        "Lesion_NSD": nsd100,
        "DSC_gain": dsc100 - ANCHOR_DSC,
        "NSD_gain": nsd100 - ANCHOR_NSD,
    }
    atomic_json(
        e100_analysis,
        {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "COMPLETE",
            "case_count": 50,
            "selection_policy": "public_DSC_qualified_p70_STUNet_or_FreeTumor_anchor_confirmation",
            "metrics": {
                "anchor": {"Lesion_DSC": ANCHOR_DSC, "Lesion_NSD": ANCHOR_NSD},
                E100_DEPLOYMENT_CONFIGURATION: {
                    "Lesion_DSC": dsc100,
                    "Lesion_NSD": nsd100,
                },
            },
            "deployments": [deployment100],
            "best_deployment": deployment100,
            "best_unconstrained_deployment": deployment100,
            "fixed_public_DSC_gate": DSC_GATE,
            "fixed_public_NSD_floor": NSD_FLOOR,
            "public_gate": True,
            "decision": "QUALIFY_SUBMISSION_PREPARATION",
            "role": "fixed_public_ensemble_selection_only",
            "source_screen": asset(e100_screen),
            "source_exact_metrics": asset(e100_metrics),
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        },
    )

    candidate100 = {
        "branch": "base_global_box",
        "checkpoint_epoch": 100,
        **deployment100,
        "checkpoint": asset(e100_checkpoint),
        "analysis": asset(e100_analysis),
    }
    candidates = [candidate100]
    if e130.get("decision") == "QUALIFY_SUBMISSION_PREPARATION":
        candidate130 = {
            "branch": e130["selected_branch"],
            "checkpoint_epoch": 130,
            "configuration": e130["selected_configuration"],
            "Lesion_DSC": float(e130["selected_fixed_public_DSC"]),
            "Lesion_NSD": float(e130["selected_fixed_public_NSD"]),
            "checkpoint": e130["checkpoint"],
            "analysis": e130["selected_analysis"],
        }
        if (
            candidate130["Lesion_DSC"] < DSC_GATE
            or candidate130["Lesion_NSD"] < NSD_FLOOR
        ):
            raise ValueError("epoch-130 selected candidate missed gates")
        candidates.append(candidate130)
    elif e130.get("decision") != "KEEP_TOUCHCOMP":
        raise ValueError("invalid epoch-130 decision")

    selected = max(
        candidates,
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.tsv"
    fields = [
        "branch",
        "checkpoint_epoch",
        "configuration",
        "Lesion_DSC",
        "Lesion_NSD",
        "DSC_gain",
        "NSD_gain",
    ]
    rows = [
        {
            "branch": "base_global_box",
            "checkpoint_epoch": 100,
            **deployment100,
        }
    ]
    rows.extend(rows130)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fields} for row in rows)

    terminal = output_dir / "TERMINAL.json"
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "QUALIFY_SUBMISSION_PREPARATION",
        "predeclared_epochs": [100, 130],
        "selection_policy": "maximum fixed-public DSC among exact gate-qualified epoch-100 and epoch-130 candidates",
        "selected_branch": selected["branch"],
        "selected_epoch": selected["checkpoint_epoch"],
        "selected_configuration": selected["configuration"],
        "selected_fixed_public_DSC": selected["Lesion_DSC"],
        "selected_fixed_public_NSD": selected["Lesion_NSD"],
        "checkpoint": selected["checkpoint"],
        "selected_analysis": selected["analysis"],
        "fixed_public_DSC_gate": DSC_GATE,
        "fixed_public_NSD_floor": NSD_FLOOR,
        "summary": asset(summary),
        "source_epoch100_screen": asset(e100_screen),
        "source_epoch100_metrics": asset(e100_metrics),
        "source_epoch130_terminal": asset(e130_terminal),
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(terminal, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e100-screen", required=True, type=Path)
    parser.add_argument("--e100-metrics", required=True, type=Path)
    parser.add_argument("--e100-checkpoint", required=True, type=Path)
    parser.add_argument("--e130-terminal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = select_best(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
