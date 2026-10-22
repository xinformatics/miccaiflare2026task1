#!/usr/bin/env python3
"""Audit BiomedParse slice-existence confidence on GF connected components.

This is a public-label diagnostic. It never modifies predictions and must not
be used to learn a case-specific hidden-validation decision rule. Its purpose
is to decide whether a small, prespecified confidence-rescue family is worth
an exact, locked-cohort evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage


def nifti_array(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image) > 0, image.GetSpacing()


def existence(npz: Path) -> np.ndarray:
    with np.load(npz, allow_pickle=False) as archive:
        values = np.asarray(archive["object_existence"], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(f"{npz}: expected one-prompt existence array, got {values.shape}")
    return values[0]


def named_directories(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"focus support must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate focus support name: {name!r}")
        result[name] = Path(raw_path)
    return result


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--whole-npz", type=Path, required=True)
    parser.add_argument("--tumor-npz", type=Path, required=True)
    parser.add_argument("--whole-support", type=Path, required=True)
    parser.add_argument("--tumor-support", type=Path, required=True)
    parser.add_argument(
        "--focus-support", action="append", default=[], metavar="NAME=PATH",
        help="optional higher-threshold GF masks used to audit component stability",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    focus_supports = named_directories(args.focus_support)

    rows: list[dict[str, object]] = []
    files = sorted(args.base.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"no masks in {args.base}")
    for index, prediction_path in enumerate(files, 1):
        case = prediction_path.name.removesuffix(".nii.gz")
        prediction, spacing = nifti_array(prediction_path)
        target, _ = nifti_array(args.labels / prediction_path.name)
        whole_support, _ = nifti_array(args.whole_support / prediction_path.name)
        tumor_support, _ = nifti_array(args.tumor_support / prediction_path.name)
        stable_masks = {
            name: nifti_array(directory / prediction_path.name)[0]
            for name, directory in focus_supports.items()
        }
        whole_existence = existence(args.whole_npz / f"{case}_0000.npz")
        tumor_existence = existence(args.tumor_npz / f"{case}_0000.npz")
        if whole_existence.shape[0] != prediction.shape[0] or tumor_existence.shape[0] != prediction.shape[0]:
            raise ValueError(f"{case}: existence depth does not match prediction")

        components, count = ndimage.label(prediction, structure=np.ones((3, 3, 3), dtype=bool))
        voxel_mm3 = float(np.prod(spacing))
        for component_id in range(1, count + 1):
            component = components == component_id
            z = np.flatnonzero(component.any(axis=(1, 2)))
            whole_scores = whole_existence[z]
            tumor_scores = tumor_existence[z]
            overlap_voxels = int(np.count_nonzero(component & target))
            row: dict[str, object] = {
                    "case": case,
                    "component": component_id,
                    "volume_mm3": float(component.sum()) * voxel_mm3,
                    "target_overlap_voxels": overlap_voxels,
                    "is_true_component": overlap_voxels > 0,
                    "whole_support": bool(np.any(component & whole_support)),
                    "tumor_support": bool(np.any(component & tumor_support)),
                    "or_support": bool(np.any(component & (whole_support | tumor_support))),
                    "whole_existence_max": float(whole_scores.max()),
                    "whole_existence_mean": float(whole_scores.mean()),
                    "tumor_existence_max": float(tumor_scores.max()),
                    "tumor_existence_mean": float(tumor_scores.mean()),
                    "or_existence_max": float(max(whole_scores.max(), tumor_scores.max())),
                    "or_existence_mean": float(max(whole_scores.mean(), tumor_scores.mean())),
                    "z_fraction": float((z.mean() + 0.5) / prediction.shape[0]),
            }
            row.update(
                {
                    f"focus_support_{name}": bool(np.any(component & mask))
                    for name, mask in stable_masks.items()
                }
            )
            rows.append(row)
        print(f"[{index:03d}/{len(files):03d}] {case}", flush=True)

    frame = pd.DataFrame(rows)
    unsupported = frame.loc[~frame["or_support"]].copy()
    labels = unsupported["is_true_component"].to_numpy(dtype=bool)
    signals = [
        "whole_existence_max",
        "whole_existence_mean",
        "tumor_existence_max",
        "tumor_existence_mean",
        "or_existence_max",
        "or_existence_mean",
    ]
    threshold_rows: list[dict[str, object]] = []
    for signal in signals:
        scores = unsupported[signal].to_numpy(dtype=np.float64)
        for threshold in (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99):
            selected = scores >= threshold
            threshold_rows.append(
                {
                    "signal": signal,
                    "threshold": threshold,
                    "rescued_true": int(np.count_nonzero(selected & labels)),
                    "rescued_false": int(np.count_nonzero(selected & ~labels)),
                    "true_total": int(labels.sum()),
                    "false_total": int((~labels).sum()),
                }
            )

    report = {
        "warning": "PUBLIC-LABEL DIAGNOSTIC ONLY; do not fit hidden case-specific decisions.",
        "cases": len(files),
        "components": len(frame),
        "true_components": int(frame["is_true_component"].sum()),
        "false_components": int((~frame["is_true_component"]).sum()),
        "unsupported_components": len(unsupported),
        "unsupported_true_components": int(labels.sum()),
        "unsupported_false_components": int((~labels).sum()),
        "auc_on_unsupported": {
            signal: auc(labels, unsupported[signal].to_numpy(dtype=np.float64)) for signal in signals
        },
        "focus_stability_on_unsupported": {
            name: {
                "rescued_true": int(
                    unsupported.loc[unsupported[f"focus_support_{name}"], "is_true_component"].sum()
                ),
                "rescued_false": int(
                    (~unsupported.loc[unsupported[f"focus_support_{name}"], "is_true_component"]).sum()
                ),
            }
            for name in focus_supports
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "components.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(args.output / "thresholds.csv", index=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "PUBLIC_LABEL_DIAGNOSTIC_DO_NOT_SUBMIT.txt").write_text(
        "This audit uses public ground truth. It diagnoses a feature family only; its outputs are not predictions.\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
