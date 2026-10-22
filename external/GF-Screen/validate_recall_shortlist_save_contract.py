#!/usr/bin/env python3
"""Fail-closed contract between recall-public selection and hidden inference."""

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
VARIANTS = {"deployed500", "deployed1000", "touching"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def require_asset(identity: object, path: Path, description: str) -> None:
    if not isinstance(identity, dict):
        raise ValueError(f"{description} identity is missing")
    if (
        Path(str(identity.get("path", ""))).resolve() != path.resolve()
        or identity.get("sha256") != sha256(path)
    ):
        raise ValueError(f"{description} identity mismatch")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_save_contract(
    public_terminal_path: Path,
    shortlist_path: Path,
    summary_path: Path,
    run_dir: Path,
    children_root: Path,
    output_path: Path,
) -> dict:
    public = json.loads(public_terminal_path.read_text(encoding="utf-8"))
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    if public.get("status") != "COMPLETE" or public.get("decision") not in {
        "KEEP_TOUCHCOMP",
        "QUALIFY_SUBMISSION_PREPARATION",
    }:
        raise ValueError("public recall-shortlist terminal is not safely complete")
    require_asset(public.get("internal_shortlist"), shortlist_path, "shortlist")
    require_asset(public.get("summary"), summary_path, "public summary")
    candidates = shortlist.get("candidates", [])
    if (
        shortlist.get("status") != "COMPLETE"
        or shortlist.get("promotion_authority") != "fixed_public_50_only"
        or shortlist.get("selection_objective")
        != "case_balanced_dice_plus_ge3000mm3_and_voxel_recall"
        or shortlist.get("large_component_minimum_mm3") != 3000
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 3
        or shortlist.get("candidate_count") != len(candidates)
        or shortlist.get("public_read") is not False
        or shortlist.get("hidden_read") is not False
        or shortlist.get("submission_created") is not False
    ):
        raise ValueError("internal recall-shortlist contract mismatch")
    allowed_roles = {
        "best_internal_dice",
        "best_large_component_recall_ge3000_within_dice_guard",
        "best_voxel_recall_within_dice_guard",
    }
    if any(
        not candidate.get("selection_roles")
        or not set(candidate["selection_roles"]) <= allowed_roles
        for candidate in candidates
    ):
        raise ValueError("internal recall-shortlist selection role mismatch")
    if (
        public.get("public_read") is not True
        or public.get("hidden_read") is not False
        or public.get("submission_created") is not False
        or public.get("uploaded") is not False
    ):
        raise ValueError("public selector provenance mismatch")
    if (
        float(public.get("fixed_public_DSC_gate", -1)) != DSC_GATE
        or float(public.get("fixed_public_NSD_floor", -1)) != NSD_FLOOR
    ):
        raise ValueError("public selector gate mismatch")
    epochs = [int(candidate["epoch"]) for candidate in candidates]
    if public.get("screened_epochs") != epochs:
        raise ValueError("public selector/shortlist epoch mismatch")

    base = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "public_terminal": asset(public_terminal_path),
        "public_summary": asset(summary_path),
        "internal_shortlist": asset(shortlist_path),
        "public_read": True,
        "hidden_read": False,
        "hidden_labels_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    if public["decision"] == "KEEP_TOUCHCOMP":
        if any(
            public.get(key) is not None
            for key in (
                "selected_epoch",
                "selected_variant",
                "checkpoint",
                "selected_public_terminal",
                "selected_predictions",
            )
        ):
            raise ValueError("nonqualified public selector retained a selected asset")
        payload = {
            **base,
            "status": "SKIP",
            "decision": "KEEP_TOUCHCOMP",
            "reason": "no recall-shortlisted checkpoint passed the fixed-public gate",
            "selected_epoch": None,
            "selected_variant": None,
            "minimum_component_mm3": None,
            "checkpoint": None,
            "selected_public_child": None,
            "selected_fixed_public_DSC": None,
            "selected_fixed_public_NSD": None,
        }
        atomic_json(output_path, payload)
        return payload

    epoch = int(public.get("selected_epoch", -1))
    variant = public.get("selected_variant")
    if epoch not in epochs or variant not in VARIANTS:
        raise ValueError("qualified public selection is outside the shortlist")
    candidate = candidates[epochs.index(epoch)]
    checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    if candidate.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("selected shortlist/checkpoint identity mismatch")
    require_asset(public.get("checkpoint"), checkpoint, "selected checkpoint")
    child_terminal = children_root / f"epoch_{epoch:04d}" / "TERMINAL.json"
    require_asset(
        public.get("selected_public_terminal"),
        child_terminal,
        "selected public child",
    )
    child = json.loads(child_terminal.read_text(encoding="utf-8"))
    if (
        child.get("status") != "COMPLETE"
        or child.get("decision") != "QUALIFY_SUBMISSION_PREPARATION"
        or child.get("selected_variant") != variant
        or child.get("hidden_read") is not False
        or child.get("submission_created") is not False
        or child.get("uploaded") is not False
    ):
        raise ValueError("selected public child contract mismatch")
    require_asset(child.get("checkpoint"), checkpoint, "child checkpoint")

    with summary_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [
        row
        for row in rows
        if int(row["checkpoint_epoch"]) == epoch and row["variant"] == variant
    ]
    if len(selected) != 1:
        raise ValueError("selected public summary row mismatch")
    row = selected[0]
    dsc, nsd = float(row["DSC"]), float(row["NSD"])
    if row["gate"] != "PROMOTE" or dsc < DSC_GATE or nsd < NSD_FLOOR:
        raise ValueError("selected public summary row missed the gate")
    if (
        float(public.get("selected_fixed_public_DSC", -1)) != dsc
        or float(public.get("selected_fixed_public_NSD", -1)) != nsd
        or Path(str(public.get("selected_predictions", ""))).resolve()
        != Path(row["predictions"]).resolve()
    ):
        raise ValueError("selected public metrics/predictions mismatch")
    minimum = 500 if variant == "deployed500" else 1000
    payload = {
        **base,
        "status": "PASS",
        "decision": "GENERATE_HIDDEN_ARCHIVE",
        "reason": "recall-shortlisted checkpoint passed the immutable fixed-public gate",
        "selected_epoch": epoch,
        "selected_variant": variant,
        "minimum_component_mm3": minimum,
        "checkpoint": asset(checkpoint),
        "selected_public_child": asset(child_terminal),
        "selected_fixed_public_DSC": dsc,
        "selected_fixed_public_NSD": nsd,
    }
    atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-terminal", required=True, type=Path)
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--children-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_save_contract(
        args.public_terminal.resolve(),
        args.shortlist.resolve(),
        args.summary.resolve(),
        args.run_dir.resolve(),
        args.children_root.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
