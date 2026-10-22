#!/usr/bin/env python3
"""Hash-bind a long-Glance public decision before hidden inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


OFFSETS = (10, 20, 30)
THRESHOLDS = (0.10, 0.20, 0.35)
VARIANTS = {"deployed500", "deployed1000", "touching"}
DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def require_public_provenance(payload: dict[str, object], description: str) -> None:
    if (
        payload.get("hidden_read") is not False
        or payload.get("submission_created") is not False
        or payload.get("uploaded") is not False
    ):
        raise ValueError(f"{description} public-only provenance mismatch")


def threshold_tag(value: float) -> str:
    return f"g{value:.2f}".replace(".", "p")


def validate_handoff(
    controller_path: Path,
    public_path: Path,
    source_path: Path,
    recipe_path: Path,
    summary_path: Path,
    run_dir: Path,
    trajectory_root: Path,
) -> dict[str, object]:
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    public = json.loads(public_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if controller.get("status") != "COMPLETE" or controller.get("decision") not in {
        "QUALIFY_SUBMISSION_PREPARATION",
        "KEEP_TOUCHCOMP",
    }:
        raise ValueError("long Glance controller is not a valid complete decision")
    if public.get("status") != "COMPLETE" or public.get("decision") != controller.get("decision"):
        raise ValueError("controller/public decision mismatch")
    require_asset(controller.get("source"), source_path, "controller source")
    require_asset(controller.get("training_recipe"), recipe_path, "controller recipe")
    require_asset(controller.get("public_terminal"), public_path, "controller public terminal")
    require_public_provenance(controller, "controller")
    require_public_provenance(public, "trajectory")
    require_asset(public.get("source"), source_path, "trajectory source")
    require_asset(public.get("summary"), summary_path, "trajectory summary")
    if source.get("status") != "PASS":
        raise ValueError("source contract did not pass")
    source_checkpoint = Path(str(source.get("source_checkpoint", {}).get("path", "")))
    require_asset(source.get("source_checkpoint"), source_checkpoint, "source checkpoint")
    source_epoch = int(source.get("source_epoch", -1))
    expected_epochs = [source_epoch + offset for offset in OFFSETS]
    if public.get("source_epoch") != source_epoch:
        raise ValueError("trajectory source epoch mismatch")
    if public.get("predeclared_offsets") != list(OFFSETS):
        raise ValueError("trajectory checkpoint offsets mismatch")
    if public.get("predeclared_epochs") != expected_epochs:
        raise ValueError("trajectory checkpoint epochs mismatch")
    if public.get("predeclared_glance_thresholds") != list(THRESHOLDS):
        raise ValueError("trajectory threshold grid mismatch")
    if abs(float(public.get("fixed_public_DSC_gate", -1)) - DSC_GATE) > 1e-12:
        raise ValueError("trajectory DSC gate mismatch")
    if abs(float(public.get("fixed_public_NSD_floor", -1)) - NSD_FLOOR) > 1e-12:
        raise ValueError("trajectory NSD floor mismatch")
    if recipe.get("predeclared_checkpoint_offsets") != list(OFFSETS):
        raise ValueError("recipe checkpoint offsets mismatch")
    if recipe.get("predeclared_glance_thresholds") != list(THRESHOLDS):
        raise ValueError("recipe threshold grid mismatch")
    if int(recipe.get("start_epoch", -1)) != source_epoch or int(recipe.get("max_epochs", -1)) != source_epoch + 30:
        raise ValueError("recipe epoch span mismatch")
    if recipe.get("public_labels_used_for_gradient") is not False:
        raise ValueError("recipe gradient provenance mismatch")

    with summary_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 27:
        raise ValueError("trajectory summary must contain exactly 27 rows")
    expected_grid = {
        (epoch, threshold, variant)
        for epoch in expected_epochs
        for threshold in THRESHOLDS
        for variant in VARIANTS
    }
    observed_grid = {
        (int(row["checkpoint_epoch"]), float(row["glance_threshold"]), row["variant"])
        for row in rows
    }
    if observed_grid != expected_grid:
        raise ValueError("trajectory summary grid mismatch")
    eligible = [
        row
        for row in rows
        if row["gate"] == "PROMOTE"
        and float(row["DSC"]) >= DSC_GATE
        and float(row["NSD"]) >= NSD_FLOOR
    ]
    eligible.sort(
        key=lambda row: (float(row["DSC"]), float(row["NSD"])), reverse=True
    )
    if controller["decision"] == "KEEP_TOUCHCOMP":
        if eligible:
            raise ValueError("KEEP trajectory contains an eligible state")
        for key in (
            "selected_epoch",
            "selected_glance_threshold",
            "selected_variant",
            "checkpoint",
            "selected_public_terminal",
        ):
            if public.get(key) is not None:
                raise ValueError(f"KEEP trajectory unexpectedly selected {key}")
        return {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "PASS",
            "decision": "KEEP_TOUCHCOMP",
            "controller": asset(controller_path),
            "public_terminal": asset(public_path),
            "source": asset(source_path),
            "training_recipe": asset(recipe_path),
            "summary": asset(summary_path),
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }

    epoch = int(public.get("selected_epoch", -1))
    threshold = float(public.get("selected_glance_threshold", -1))
    variant = str(public.get("selected_variant", ""))
    if epoch not in expected_epochs or threshold not in THRESHOLDS or variant not in VARIANTS:
        raise ValueError("selected state is outside the predeclared grid")
    checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    require_asset(public.get("checkpoint"), checkpoint, "selected checkpoint")
    child = trajectory_root / f"epoch_{epoch:04d}" / threshold_tag(threshold)
    child_terminal = child / "TERMINAL.json"
    child_summary = child / "summary.tsv"
    require_asset(public.get("selected_public_terminal"), child_terminal, "selected child terminal")
    child_payload = json.loads(child_terminal.read_text(encoding="utf-8"))
    if (
        child_payload.get("status") != "COMPLETE"
        or child_payload.get("decision") != "QUALIFY_SUBMISSION_PREPARATION"
        or child_payload.get("selected_variant") != variant
        or abs(float(child_payload.get("glance_threshold", -1)) - threshold) > 1e-12
    ):
        raise ValueError("selected child state mismatch")
    require_asset(child_payload.get("checkpoint"), checkpoint, "selected child checkpoint")
    require_asset(child_payload.get("summary"), child_summary, "selected child summary")
    require_public_provenance(child_payload, "selected child")

    selected = [
        row
        for row in rows
        if int(row["checkpoint_epoch"]) == epoch
        and abs(float(row["glance_threshold"]) - threshold) <= 1e-12
        and row["variant"] == variant
    ]
    if len(selected) != 1:
        raise ValueError("selected trajectory summary row mismatch")
    row = selected[0]
    if not eligible or eligible[0] != row:
        raise ValueError("selected trajectory row is not the best eligible state")
    dsc, nsd = float(row["DSC"]), float(row["NSD"])
    if row["gate"] != "PROMOTE" or dsc < DSC_GATE or nsd < NSD_FLOOR:
        raise ValueError("selected trajectory row missed the public gate")
    if (
        abs(float(public.get("selected_fixed_public_DSC", -1)) - dsc) > 1e-12
        or abs(float(public.get("selected_fixed_public_NSD", -1)) - nsd) > 1e-12
    ):
        raise ValueError("selected trajectory metrics mismatch")
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "decision": "QUALIFY_SUBMISSION_PREPARATION",
        "selected_epoch": epoch,
        "selected_glance_threshold": threshold,
        "selected_variant": variant,
        "selected_fixed_public_DSC": dsc,
        "selected_fixed_public_NSD": nsd,
        "checkpoint": asset(checkpoint),
        "selected_public_terminal": asset(child_terminal),
        "controller": asset(controller_path),
        "public_terminal": asset(public_path),
        "source": asset(source_path),
        "training_recipe": asset(recipe_path),
        "summary": asset(summary_path),
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_handoff(
        args.controller,
        args.public,
        args.source,
        args.recipe,
        args.summary,
        args.run_dir,
        args.trajectory_root,
    )
    atomic_json(args.output, payload)
    print(payload["decision"])


if __name__ == "__main__":
    main()
