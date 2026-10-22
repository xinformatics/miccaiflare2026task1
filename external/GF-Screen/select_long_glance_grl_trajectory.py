#!/usr/bin/env python3
"""Validate and select one fixed-public state from the long Glance trajectory."""

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


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def threshold_tag(value: float) -> str:
    return f"g{value:.2f}".replace(".", "p")


def select_trajectory(
    run_dir: Path,
    source_path: Path,
    root: Path,
    summary_path: Path,
    terminal_path: Path,
    lesion_error_summary_path: Path | None = None,
) -> dict[str, object]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "PASS":
        raise ValueError("Glance source contract did not pass")
    source_epoch = int(source["source_epoch"])
    source_checkpoint = Path(str(source["source_checkpoint"]["path"]))
    require_asset(
        source.get("source_checkpoint"), source_checkpoint, "source checkpoint"
    )
    expected_epochs = tuple(source_epoch + offset for offset in OFFSETS)

    rows_out: list[dict[str, str]] = []
    children: list[dict[str, object]] = []
    fieldnames: list[str] | None = None
    for epoch in expected_epochs:
        checkpoint = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        if not checkpoint.is_file():
            raise ValueError(f"missing trajectory checkpoint: {checkpoint}")
        for threshold in THRESHOLDS:
            child = root / f"epoch_{epoch:04d}" / threshold_tag(threshold)
            child_terminal = child / "TERMINAL.json"
            child_summary = child / "summary.tsv"
            payload = json.loads(child_terminal.read_text(encoding="utf-8"))
            if payload.get("status") != "COMPLETE" or payload.get("decision") not in {
                "QUALIFY_SUBMISSION_PREPARATION",
                "KEEP_TOUCHCOMP",
            }:
                raise ValueError(
                    f"epoch {epoch}, threshold {threshold}: invalid child terminal"
                )
            require_asset(
                payload.get("checkpoint"), checkpoint,
                f"epoch {epoch}, threshold {threshold} checkpoint",
            )
            require_asset(
                payload.get("summary"), child_summary,
                f"epoch {epoch}, threshold {threshold} summary",
            )
            if abs(float(payload.get("glance_threshold", -1)) - threshold) > 1e-12:
                raise ValueError(
                    f"epoch {epoch}, threshold {threshold}: Glance threshold mismatch"
                )
            if (
                payload.get("hidden_read") is not False
                or payload.get("submission_created") is not False
                or payload.get("uploaded") is not False
            ):
                raise ValueError(
                    f"epoch {epoch}, threshold {threshold}: public-only provenance mismatch"
                )
            with child_summary.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if len(rows) != 3 or {row["variant"] for row in rows} != VARIANTS:
                raise ValueError(
                    f"epoch {epoch}, threshold {threshold}: deployment coverage mismatch"
                )
            if fieldnames is None:
                fieldnames = ["checkpoint_epoch", "glance_threshold", *rows[0].keys()]
            for row in rows:
                rows_out.append(
                    {
                        "checkpoint_epoch": str(epoch),
                        "glance_threshold": f"{threshold:.2f}",
                        **row,
                    }
                )
            children.append(
                {
                    "epoch": epoch,
                    "glance_threshold": threshold,
                    "checkpoint": asset(checkpoint),
                    "terminal": asset(child_terminal),
                    "summary": asset(child_summary),
                }
            )

    assert fieldnames is not None
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_name(
        f".{summary_path.name}.{os.getpid()}.tmp"
    )
    with temporary_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)
    os.replace(temporary_summary, summary_path)

    eligible = [
        row
        for row in rows_out
        if row["gate"] == "PROMOTE"
        and float(row["DSC"]) >= DSC_GATE
        and float(row["NSD"]) >= NSD_FLOOR
    ]
    eligible.sort(
        key=lambda row: (float(row["DSC"]), float(row["NSD"])), reverse=True
    )
    selected = eligible[0] if eligible else None
    selected_epoch = int(selected["checkpoint_epoch"]) if selected else None
    selected_checkpoint = (
        run_dir / f"checkpoint_epoch_{selected_epoch:04d}.pt"
        if selected
        else None
    )
    selected_threshold = (
        float(selected["glance_threshold"]) if selected else None
    )
    selected_child = (
        root
        / f"epoch_{selected_epoch:04d}"
        / threshold_tag(selected_threshold)
        / "TERMINAL.json"
        if selected
        else None
    )
    best_dsc = sorted(
        rows_out,
        key=lambda row: (float(row["DSC"]), float(row["NSD"])),
        reverse=True,
    )[0]
    lesion_error_identity = None
    if lesion_error_summary_path is not None:
        diagnostic = json.loads(
            lesion_error_summary_path.read_text(encoding="utf-8")
        )
        diagnostic_predictions = Path(
            str(diagnostic.get("configuration", {}).get("predictions", ""))
        )
        if (
            int(diagnostic.get("cases", -1)) != 50
            or diagnostic_predictions.resolve()
            != Path(best_dsc["predictions"]).resolve()
        ):
            raise ValueError("best-DSC lesion-error diagnostic identity mismatch")
        lesion_error_identity = asset(lesion_error_summary_path)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": (
            "QUALIFY_SUBMISSION_PREPARATION" if selected else "KEEP_TOUCHCOMP"
        ),
        "reason": (
            "best predeclared long-Glance state passed the fixed-public gate"
            if selected
            else "no predeclared long-Glance state passed the fixed-public gate"
        ),
        "source": asset(source_path),
        "source_epoch": source_epoch,
        "predeclared_offsets": list(OFFSETS),
        "predeclared_epochs": list(expected_epochs),
        "predeclared_glance_thresholds": list(THRESHOLDS),
        "selected_epoch": selected_epoch,
        "selected_glance_threshold": selected_threshold,
        "selected_variant": selected["variant"] if selected else None,
        "selected_fixed_public_DSC": float(selected["DSC"]) if selected else None,
        "selected_fixed_public_NSD": float(selected["NSD"]) if selected else None,
        "checkpoint": asset(selected_checkpoint) if selected_checkpoint else None,
        "selected_public_terminal": asset(selected_child) if selected_child else None,
        "best_dsc_diagnostic": {
            "checkpoint_epoch": int(best_dsc["checkpoint_epoch"]),
            "glance_threshold": float(best_dsc["glance_threshold"]),
            "variant": best_dsc["variant"],
            "DSC": float(best_dsc["DSC"]),
            "NSD": float(best_dsc["NSD"]),
            "lesion_error_summary": lesion_error_identity,
            "selection_role": "diagnostic_only_not_used_for_promotion",
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
    atomic_json(terminal_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--lesion-error-summary", type=Path)
    args = parser.parse_args()
    payload = select_trajectory(
        args.run_dir,
        args.source,
        args.trajectory_root,
        args.summary,
        args.terminal,
        args.lesion_error_summary,
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "decision",
                    "selected_epoch",
                    "selected_glance_threshold",
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
