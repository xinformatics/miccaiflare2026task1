#!/usr/bin/env python3
"""Verify raw public CTs reproduce the frozen BiomedParse input tensors.

This verifier reads no labels and performs no model inference. It proves the
raw-NIfTI-to-uint8 boundary used by Docker is identical to the input artifacts
that produced the immutable winner prompt caches.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk


PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "external/BiomedParse"))
from miccai_nifti_adapter import case_name, window_to_uint8  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        type=Path,
        default=PROJECT / "validation/Validation-Public-Images",
    )
    parser.add_argument(
        "--reference-inputs",
        type=Path,
        default=PROJECT / "external/BiomedParse/public_a160_240_inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT
        / "external/GF-Screen/docker_submission/analysis/biomedparse_raw_adapter_public_equivalence.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = sorted(args.images.glob("*.nii.gz"))
    if len(sources) != 50:
        raise RuntimeError(f"expected exactly 50 public CTs, got {len(sources)}")
    digest = hashlib.sha256()
    rows = []
    mismatches = []
    for index, source in enumerate(sources, 1):
        reference_path = args.reference_inputs / f"{case_name(source)}.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        image = sitk.ReadImage(str(source))
        observed = window_to_uint8(sitk.GetArrayFromImage(image), -160.0, 240.0)
        observed_spacing = np.asarray(image.GetSpacing(), dtype=np.float32)[::-1]
        with np.load(reference_path, allow_pickle=False) as archive:
            expected = np.asarray(archive["imgs"])
            expected_spacing = np.asarray(archive["spacing"])
        image_equal = bool(np.array_equal(observed, expected))
        spacing_equal = bool(np.array_equal(observed_spacing, expected_spacing))
        if not image_equal or not spacing_equal:
            mismatches.append(source.name)
        digest.update(source.name.encode("utf-8"))
        digest.update(np.asarray(observed.shape, dtype=np.int64).tobytes())
        digest.update(observed.tobytes(order="C"))
        rows.append(
            {
                "case": source.name,
                "shape_zyx": list(observed.shape),
                "image_exact": image_equal,
                "spacing_exact": spacing_equal,
                "minimum": int(observed.min()),
                "maximum": int(observed.max()),
            }
        )
        print(f"verified {index}/50 {source.name}: exact={image_equal and spacing_equal}", flush=True)
    exact = not mismatches
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE" if exact else "FAILED",
        "decision": "QUALIFY_RAW_BIOMEDPARSE_ADAPTER" if exact else "REJECT_RAW_BIOMEDPARSE_ADAPTER",
        "case_count": 50,
        "window_hu": [-160.0, 240.0],
        "rounding": "numpy_rint_then_uint8",
        "exact": exact,
        "mismatches": mismatches,
        "aggregate_input_sha256": digest.hexdigest(),
        "cases": rows,
        "public_images_read": True,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: payload[key] for key in ("decision", "case_count", "aggregate_input_sha256")}, indent=2))
    if not exact:
        raise SystemExit("raw BiomedParse adapter equivalence failed")


if __name__ == "__main__":
    main()
