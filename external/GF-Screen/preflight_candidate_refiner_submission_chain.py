#!/usr/bin/env python3
"""Fail-closed inventory and provenance audit before hidden prediction packaging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty file: {path}")


def inventory(directory: Path, suffix: str, strip_suffix: str, expected: int) -> dict:
    if not directory.is_dir():
        raise SystemExit(f"missing directory: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name.endswith(suffix))
    if len(files) != expected:
        raise SystemExit(f"expected {expected} {suffix} files in {directory}, found {len(files)}")
    empty = [path.name for path in files if path.stat().st_size <= 0]
    if empty:
        raise SystemExit(f"empty files in {directory}: {empty[:3]}")
    cases = [path.name[: -len(strip_suffix)] for path in files]
    if len(set(cases)) != expected:
        raise SystemExit(f"duplicate normalized cases in {directory}")
    names_hash = hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest()
    return {"path": str(directory.resolve()), "count": expected, "cases": cases, "case_names_sha256": names_hash}


def zip_inventory(path: Path, expected: int) -> dict:
    require_file(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        bad_layout = [item.filename for item in infos if "/" in item.filename or not item.filename.endswith(".nii.gz")]
        if bad_layout:
            raise SystemExit(f"ZIP is not flat .nii.gz-only: {bad_layout[:3]}")
        names = [item.filename for item in infos]
        if len(names) != expected or len(set(names)) != expected:
            raise SystemExit(f"expected {expected} unique ZIP members, found {len(names)}")
        empty = [item.filename for item in infos if item.file_size <= 0]
        if empty:
            raise SystemExit(f"empty ZIP members: {empty[:3]}")
        first_bad = archive.testzip()
        if first_bad is not None:
            raise SystemExit(f"ZIP CRC failure: {first_bad}")
    cases = sorted(name[: -len(".nii.gz")] for name in names)
    return {
        "path": str(path.resolve()),
        "count": expected,
        "cases": cases,
        "case_names_sha256": hashlib.sha256(("\n".join(cases) + "\n").encode()).hexdigest(),
        "sha256": sha256_file(path),
        "crc": "PASS",
        "flat": True,
    }


def load_json(path: Path) -> dict:
    require_file(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def audit_cells(path: Path, expected: int) -> dict:
    value = load_json(path)
    checks = {
        "cases": value.get("cases") == expected,
        "sources": value.get("sources") == 8,
        "method": value.get("method") == "filled_48mm_occupied_lung_cells_in_eight_parity_sources",
        "target_spacing_mm": value.get("parameters", {}).get("target_spacing_mm") == 2.5,
        "cell_stride_mm": value.get("parameters", {}).get("cell_stride_mm") == 48.0,
        "maximum_extent": float(value.get("maximum_component_axis_extent_mm", float("inf"))) <= 48.0001,
        "labels_read": value.get("labels_read") is False,
        "complete_labels_required": value.get("complete_labels_required") is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"invalid lung-cell marker {path}: {checks}")
    rows = value.get("rows", [])
    row_cases = sorted(str(row.get("case", ""))[: -len(".nii.gz")] for row in rows)
    if len(rows) != expected or len(set(row_cases)) != expected:
        raise SystemExit(f"invalid lung-cell row inventory: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "checks": checks, "cases": row_cases}


def same_cases(reference: dict, candidates: dict[str, dict], cohort: str) -> None:
    wanted = set(reference["cases"])
    for name, candidate in candidates.items():
        if set(candidate["cases"]) != wanted:
            missing = sorted(wanted - set(candidate["cases"]))[:3]
            extra = sorted(set(candidate["cases"]) - wanted)[:3]
            raise SystemExit(f"{cohort} inventory mismatch for {name}: missing={missing} extra={extra}")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "public_images", "public_labels", "public_inputs", "public_epoch105", "public_currentgf",
        "public_base", "hidden_images", "hidden_inputs", "hidden_currentgf", "champion_zip",
        "public_cells", "hidden_cells", "selection", "calibration", "weak_plan", "margin_decision",
        "cell_decision", "gf_checkpoint", "packager", "public_summary", "output",
    ):
        result.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    result.add_argument(
        "--extra-asset", action="append", default=[], metavar="NAME=PATH",
        help="Additional deployment decision or implementation to SHA256-bind.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    public = {
        "images": inventory(args.public_images, "_0000.nii.gz", "_0000.nii.gz", 50),
        "labels": inventory(args.public_labels, ".nii.gz", ".nii.gz", 50),
        "inputs": inventory(args.public_inputs, "_0000.npz", "_0000.npz", 50),
        "epoch105": inventory(args.public_epoch105, ".nii.gz", ".nii.gz", 50),
        "currentgf": inventory(args.public_currentgf, ".nii.gz", ".nii.gz", 50),
        "base": inventory(args.public_base, ".nii.gz", ".nii.gz", 50),
    }
    same_cases(public["images"], {name: value for name, value in public.items() if name != "images"}, "public")

    hidden = {
        "images": inventory(args.hidden_images, "_0000.nii.gz", "_0000.nii.gz", 100),
        "inputs": inventory(args.hidden_inputs, "_0000.npz", "_0000.npz", 100),
        "currentgf": inventory(args.hidden_currentgf, ".nii.gz", ".nii.gz", 100),
        "champion_zip": zip_inventory(args.champion_zip, 100),
    }
    same_cases(hidden["images"], {name: value for name, value in hidden.items() if name != "images"}, "hidden")

    public_cells = audit_cells(args.public_cells, 50)
    hidden_cells = audit_cells(args.hidden_cells, 100)
    if set(public_cells["cases"]) != set(public["images"]["cases"]):
        raise SystemExit("public lung-cell marker does not match public images")
    if set(hidden_cells["cases"]) != set(hidden["images"]["cases"]):
        raise SystemExit("hidden lung-cell marker does not match hidden images")

    cell_decision = load_json(args.cell_decision)
    if cell_decision.get("decision") != "PROMOTE_LUNG_CELLS_707":
        raise SystemExit("lung-cell decision is not PROMOTE_LUNG_CELLS_707")
    margin = load_json(args.margin_decision)
    effective_margin = margin.get("effective_margin_mm")
    if effective_margin is None:
        effective_margin = margin.get("baseline_margin_mm") if margin.get("decision") == "KEEP_MARGIN_24" else margin.get("selected", {}).get("margin_mm")
    if int(effective_margin) not in {16, 24, 32, 48, 72, 96}:
        raise SystemExit(f"invalid frozen bounding-box margin: {effective_margin}")

    asset_paths = {
        "preflight_implementation": Path(__file__).resolve(),
        "selection": args.selection,
        "calibration": args.calibration,
        "weak_plan": args.weak_plan,
        "margin_decision": args.margin_decision,
        "cell_decision": args.cell_decision,
        "gf_checkpoint": args.gf_checkpoint,
        "packager": args.packager,
        "public_summary": args.public_summary,
    }
    assets = {}
    for name, path in asset_paths.items():
        require_file(path)
        assets[name] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)}
    for specification in args.extra_asset:
        if "=" not in specification:
            raise SystemExit(f"extra asset must use NAME=PATH: {specification}")
        name, raw_path = specification.split("=", 1)
        path = Path(raw_path)
        if not name or name in assets:
            raise SystemExit(f"invalid or duplicate extra asset: {specification}")
        require_file(path)
        assets[name] = {
            "path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "public": public,
        "hidden": hidden,
        "public_cells": public_cells,
        "hidden_cells": hidden_cells,
        "assets": assets,
        "effective_margin_mm": int(effective_margin),
        "labels_read": {"public": True, "hidden": False},
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "output": str(args.output), "public_cases": 50, "hidden_cases": 100}))


if __name__ == "__main__":
    main()
