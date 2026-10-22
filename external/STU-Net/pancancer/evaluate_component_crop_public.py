#!/usr/bin/env python3
"""Evaluate cropped-STU confirmation against full STU on fixed public cases."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GF_ROOT = ROOT / "external/GF-Screen"
for source in (Path(__file__).resolve().parent, GF_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from analyze_stunet_gf_ensemble import confirmed_anchor_components  # noqa: E402
from fast_official_lesion_metrics import score_case  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluate_case(item: tuple[str, dict[str, str]]) -> dict:
    name, rendered = item
    paths = {key: Path(value) for key, value in rendered.items()}
    anchor_image = nib.load(str(paths["anchor"] / name))
    anchor = np.asanyarray(anchor_image.dataobj).astype(bool)
    cropped_image = nib.load(str(paths["cropped"] / name))
    truth_image = nib.load(str(paths["labels"] / name))
    if (
        cropped_image.shape != anchor_image.shape
        or truth_image.shape != anchor_image.shape
        or not np.allclose(cropped_image.affine, anchor_image.affine, atol=1e-4)
        or not np.allclose(truth_image.affine, anchor_image.affine, atol=1e-4)
    ):
        raise ValueError(f"geometry mismatch: {name}")
    cropped = np.asanyarray(cropped_image.dataobj).astype(bool)
    truth = np.asanyarray(truth_image.dataobj).astype(bool)
    with np.load(paths["baseline"] / f"{name.removesuffix('.nii.gz')}.prob16.npz") as archive:
        probability = archive["lesion_probability_uint16"]
    if probability.dtype != np.uint16 or probability.shape != anchor.shape:
        raise ValueError(f"invalid baseline probability: {name}")
    full = confirmed_anchor_components(
        anchor,
        probability.astype(np.float32) / 65535.0 > 0.70,
        np.zeros_like(anchor),
        "stu",
    )
    spacing = anchor_image.header.get_zooms()[:3]
    full_score = score_case(truth.astype(np.uint8), full.astype(np.uint8), spacing)
    cropped_score = score_case(truth.astype(np.uint8), cropped.astype(np.uint8), spacing)
    intersection = int(np.count_nonzero(full & cropped))
    denominator = int(full.sum() + cropped.sum())
    return {
        "case": name,
        "mask_equal": bool(np.array_equal(full, cropped)),
        "mask_dice_vs_full": float(2 * intersection / denominator) if denominator else 1.0,
        "full_stunet_only": {"Lesion_DSC": float(full_score[0]), "Lesion_NSD": float(full_score[1])},
        "cropped_stunet_only": {"Lesion_DSC": float(cropped_score[0]), "Lesion_NSD": float(cropped_score[1])},
        "delta_DSC": float(cropped_score[0] - full_score[0]),
        "delta_NSD": float(cropped_score[1] - full_score[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", required=True, type=Path)
    parser.add_argument("--predict-terminal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    paths = {
        "anchor": ROOT / "external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions",
        "baseline": ROOT / "external/STU-Net/pancancer/public_sweeps/stunetb_matched_e160_20260826/component_predictions",
        "labels": ROOT / "validation/Validation-Public-Labels",
        "cropped": args.cropped,
    }
    label_parts = {part.casefold() for part in paths["labels"].resolve().parts}
    if "validation-public-labels" not in label_parts or "validation-hidden-labels" in label_parts:
        raise ValueError("fixed-public labels required")
    names = sorted(path.name for path in paths["labels"].glob("*.nii.gz"))
    expected = set(names)
    if len(names) != 50:
        raise RuntimeError("fixed-public inventory must contain 50 cases")
    for role in ("anchor", "cropped"):
        if {path.name for path in paths[role].glob("*.nii.gz")} != expected:
            raise RuntimeError(f"{role} inventory mismatch")
    if {
        f"{path.name.removesuffix('.prob16.npz')}.nii.gz"
        for path in paths["baseline"].glob("*.prob16.npz")
    } != expected:
        raise RuntimeError("baseline probability inventory mismatch")
    predict = json.loads(args.predict_terminal.read_text(encoding="utf-8"))
    if (
        predict.get("status") != "COMPLETE"
        or predict.get("case_count") != 50
        or predict.get("margin_mm") != 24.0
        or predict.get("public_labels_read")
        or predict.get("hidden_read")
        or predict.get("submission_created")
        or predict.get("uploaded")
    ):
        raise ValueError("invalid cropped-STU prediction terminal")
    rendered = {key: str(value.resolve()) for key, value in paths.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        cases = list(executor.map(evaluate_case, ((name, rendered) for name in names)))

    def aggregate(key: str) -> dict[str, float | int]:
        dsc = np.asarray([case[key]["Lesion_DSC"] for case in cases])
        nsd = np.asarray([case[key]["Lesion_NSD"] for case in cases])
        return {
            "Lesion_DSC": float(dsc.mean()),
            "Lesion_NSD": float(nsd.mean()),
            "zero_DSC_cases": int(np.count_nonzero(dsc == 0)),
        }

    full = aggregate("full_stunet_only")
    cropped = aggregate("cropped_stunet_only")
    cropped["delta_DSC_vs_full"] = float(cropped["Lesion_DSC"] - full["Lesion_DSC"])
    cropped["delta_NSD_vs_full"] = float(cropped["Lesion_NSD"] - full["Lesion_NSD"])
    times = np.asarray([float(row["seconds"]) for row in predict["cases"]])
    runtime = {
        "mean_seconds": float(times.mean()),
        "p95_seconds": float(np.quantile(times, 0.95)),
        "max_seconds": float(times.max()),
        "peak_gpu_reserved_mb": float(predict["peak_gpu_reserved_mb"]),
    }
    quality_pass = bool(
        cropped["Lesion_DSC"] >= full["Lesion_DSC"] - 0.005
        and cropped["Lesion_NSD"] >= full["Lesion_NSD"] - 0.005
    )
    runtime_pass = bool(runtime["max_seconds"] <= 20 and runtime["peak_gpu_reserved_mb"] <= 4096)
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "QUALIFY_CROPPED_STUNET_STAGE" if quality_pass and runtime_pass else "REJECT_OR_REVISE_CROPPED_STUNET_STAGE",
        "case_count": 50,
        "full_stunet_only": full,
        "cropped_stunet_only": cropped,
        "mask_equal_cases": int(sum(case["mask_equal"] for case in cases)),
        "mean_mask_dice_vs_full": float(np.mean([case["mask_dice_vs_full"] for case in cases])),
        "runtime": runtime,
        "quality_pass": quality_pass,
        "runtime_pass": runtime_pass,
        "cases": cases,
        "predict_terminal": {"path": str(args.predict_terminal.resolve()), "sha256": sha256(args.predict_terminal)},
        "public_images_read": True,
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "deployment_authorized": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("decision", "full_stunet_only", "cropped_stunet_only", "runtime", "quality_pass", "runtime_pass")}, indent=2))


if __name__ == "__main__":
    main()
