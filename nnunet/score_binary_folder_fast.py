#!/usr/bin/env python3
"""Memory-bounded challenge DSC/NSD scoring for a binary prediction folder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GF_ROOT = PROJECT_ROOT / "external" / "GF-Screen"
sys.path.insert(0, str(GF_ROOT))
from fast_official_lesion_metrics import (  # noqa: E402
    prepare_surface_reference,
    score_case_with_reference,
)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    if "validation-public-labels" not in {
        part.casefold() for part in args.labels.resolve().parts
    }:
        raise ValueError("scorer requires the fixed Validation-Public-Labels path")
    names = sorted(path.name for path in args.labels.glob("*.nii.gz"))
    if len(names) != 50:
        raise ValueError(f"expected 50 public labels, found {len(names)}")
    if {path.name for path in args.predictions.glob("*.nii.gz")} != set(names):
        raise ValueError("prediction inventory differs from public labels")

    rows = []
    for index, name in enumerate(names, start=1):
        label_image = nib.load(str(args.labels / name))
        prediction_image = nib.load(str(args.predictions / name))
        if prediction_image.shape != label_image.shape or not np.allclose(
            prediction_image.affine, label_image.affine, atol=1e-4
        ):
            raise ValueError(f"geometry mismatch: {name}")
        truth = np.asanyarray(label_image.dataobj).astype(bool, copy=False)
        prediction = np.asanyarray(prediction_image.dataobj)
        if not np.isin(np.unique(prediction), [0, 1]).all():
            raise ValueError(f"nonbinary prediction: {name}")
        prediction = prediction.astype(bool, copy=False)
        spacing = tuple(float(value) for value in label_image.header.get_zooms()[:3])
        reference = prepare_surface_reference(truth, spacing) if truth.any() else None
        dsc, nsd = score_case_with_reference(
            truth, prediction, spacing, reference
        )
        rows.append(
            {
                "Name": name.removesuffix(".nii.gz"),
                "Lesion_DSC": float(dsc),
                "Lesion_NSD": float(nsd),
                "pred_fg_voxels": int(prediction.sum()),
            }
        )
        print(f"metric progress {index}/50: {name} DSC={dsc:.6f} NSD={nsd:.6f}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "case_count": len(rows),
        "Lesion_DSC": float(np.mean([row["Lesion_DSC"] for row in rows])),
        "Lesion_NSD": float(np.mean([row["Lesion_NSD"] for row in rows])),
        "predictions": str(args.predictions.resolve()),
        "labels": str(args.labels.resolve()),
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

