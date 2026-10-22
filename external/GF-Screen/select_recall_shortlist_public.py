#!/usr/bin/env python3
"""Select one fixed-public winner from the recall-aware checkpoint shortlist."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


VARIANTS = {"deployed500", "deployed1000", "touching"}
DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019
SELECTION_OBJECTIVE = "case_balanced_dice_plus_ge3000mm3_and_voxel_recall"
ALLOWED_SELECTION_ROLES = {
    "best_internal_dice",
    "best_large_component_recall_ge3000_within_dice_guard",
    "best_voxel_recall_within_dice_guard",
}


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


def select_public_shortlist(
    run_dir: Path,
    shortlist_path: Path,
    children_root: Path,
    summary_path: Path,
    terminal_path: Path,
) -> dict:
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    candidates = shortlist.get("candidates")
    if (
        shortlist.get("status") != "COMPLETE"
        or shortlist.get("promotion_authority") != "fixed_public_50_only"
        or shortlist.get("selection_objective") != SELECTION_OBJECTIVE
        or shortlist.get("large_component_minimum_mm3") != 3000
        or shortlist.get("public_read") is not False
        or shortlist.get("hidden_read") is not False
        or shortlist.get("submission_created") is not False
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 3
        or int(shortlist.get("candidate_count", -1)) != len(candidates)
    ):
        raise ValueError("invalid internal recall shortlist contract")
    if (
        float(shortlist.get("fixed_public_DSC_gate", -1)) != DSC_GATE
        or float(shortlist.get("fixed_public_NSD_floor", -1)) != NSD_FLOOR
    ):
        raise ValueError("shortlist fixed-public gate mismatch")

    epochs = [int(candidate["epoch"]) for candidate in candidates]
    if len(set(epochs)) != len(epochs):
        raise ValueError("shortlist epochs must be unique")
    if any(
        not candidate.get("selection_roles")
        or not set(candidate["selection_roles"]) <= ALLOWED_SELECTION_ROLES
        for candidate in candidates
    ):
        raise ValueError("shortlist contains a deprecated selection role")
    by_epoch = {int(candidate["epoch"]): candidate for candidate in candidates}
    all_rows: list[dict[str, str]] = []
    children: list[dict] = []
    fieldnames: list[str] | None = None
    for epoch in epochs:
        candidate = by_epoch[epoch]
        checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if candidate.get("checkpoint_sha256") != sha256(checkpoint):
            raise ValueError(f"epoch {epoch}: shortlist/checkpoint hash mismatch")
        child = children_root / f"epoch_{epoch:04d}"
        child_terminal = child / "TERMINAL.json"
        child_summary = child / "summary.tsv"
        payload = json.loads(child_terminal.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("decision") not in {
            "QUALIFY_SUBMISSION_PREPARATION",
            "KEEP_TOUCHCOMP",
        }:
            raise ValueError(f"epoch {epoch}: invalid public child terminal")
        require_asset(payload.get("checkpoint"), checkpoint, f"epoch {epoch} checkpoint")
        require_asset(payload.get("summary"), child_summary, f"epoch {epoch} summary")
        if (
            payload.get("public_read") is not True
            or payload.get("hidden_read") is not False
            or payload.get("submission_created") is not False
            or payload.get("uploaded") is not False
        ):
            raise ValueError(f"epoch {epoch}: public-only provenance mismatch")
        if (
            float(payload.get("fixed_public_DSC_gate", -1)) != DSC_GATE
            or float(payload.get("fixed_public_NSD_floor", -1)) != NSD_FLOOR
        ):
            raise ValueError(f"epoch {epoch}: child gate mismatch")
        with child_summary.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 3 or {row["variant"] for row in rows} != VARIANTS:
            raise ValueError(f"epoch {epoch}: deployment coverage mismatch")
        eligible = sorted(
            (
                row
                for row in rows
                if row["gate"] == "PROMOTE"
                and float(row["DSC"]) >= DSC_GATE
                and float(row["NSD"]) >= NSD_FLOOR
            ),
            key=lambda row: (float(row["DSC"]), float(row["NSD"])),
            reverse=True,
        )
        expected_variant = eligible[0]["variant"] if eligible else None
        if payload.get("selected_variant") != expected_variant:
            raise ValueError(f"epoch {epoch}: child selection mismatch")
        if fieldnames is None:
            fieldnames = [
                "checkpoint_epoch",
                "internal_selection_roles",
                *rows[0].keys(),
            ]
        roles = ",".join(candidate.get("selection_roles", ()))
        for row in rows:
            all_rows.append(
                {
                    "checkpoint_epoch": str(epoch),
                    "internal_selection_roles": roles,
                    **row,
                }
            )
        children.append(
            {
                "epoch": epoch,
                "internal_selection_roles": candidate.get("selection_roles", []),
                "checkpoint": asset(checkpoint),
                "terminal": asset(child_terminal),
                "summary": asset(child_summary),
            }
        )

    assert fieldnames is not None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary, summary_path)

    eligible = sorted(
        (
            row
            for row in all_rows
            if row["gate"] == "PROMOTE"
            and float(row["DSC"]) >= DSC_GATE
            and float(row["NSD"]) >= NSD_FLOOR
        ),
        key=lambda row: (float(row["DSC"]), float(row["NSD"])),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    selected_epoch = int(selected["checkpoint_epoch"]) if selected else None
    selected_child = (
        children_root / f"epoch_{selected_epoch:04d}" if selected else None
    )
    selected_checkpoint = (
        run_dir / f"checkpoint_epoch_{selected_epoch:04d}.pt" if selected else None
    )
    best_dsc = max(
        all_rows, key=lambda row: (float(row["DSC"]), float(row["NSD"]))
    )
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": (
            "QUALIFY_SUBMISSION_PREPARATION" if selected else "KEEP_TOUCHCOMP"
        ),
        "reason": (
            "one recall-shortlisted checkpoint passed the immutable fixed-public gate"
            if selected
            else "no recall-shortlisted checkpoint passed the immutable fixed-public gate"
        ),
        "internal_shortlist": asset(shortlist_path),
        "screened_epochs": epochs,
        "selected_epoch": selected_epoch,
        "selected_variant": selected["variant"] if selected else None,
        "selected_fixed_public_DSC": float(selected["DSC"]) if selected else None,
        "selected_fixed_public_NSD": float(selected["NSD"]) if selected else None,
        "checkpoint": asset(selected_checkpoint) if selected_checkpoint else None,
        "selected_public_terminal": (
            asset(selected_child / "TERMINAL.json") if selected_child else None
        ),
        "selected_predictions": (
            str(Path(selected["predictions"]).resolve()) if selected else None
        ),
        "best_dsc_diagnostic": {
            "checkpoint_epoch": int(best_dsc["checkpoint_epoch"]),
            "variant": best_dsc["variant"],
            "DSC": float(best_dsc["DSC"]),
            "NSD": float(best_dsc["NSD"]),
            "selection_role": "diagnostic_only_unless_fixed_public_gate_passes",
        },
        "summary": asset(summary_path),
        "children": children,
        "fixed_public_DSC_gate": DSC_GATE,
        "fixed_public_NSD_floor": NSD_FLOOR,
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(terminal_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--shortlist", required=True, type=Path)
    parser.add_argument("--children-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    payload = select_public_shortlist(
        args.run_dir.resolve(),
        args.shortlist.resolve(),
        args.children_root.resolve(),
        args.summary.resolve(),
        args.terminal.resolve(),
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "decision",
                    "screened_epochs",
                    "selected_epoch",
                    "selected_variant",
                    "selected_fixed_public_DSC",
                    "selected_fixed_public_NSD",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
