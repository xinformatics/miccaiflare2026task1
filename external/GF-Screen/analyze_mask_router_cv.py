#!/usr/bin/env python3
"""Cross-validated, label-free routing between two deployed mask systems.

Features use only the two predicted masks and their NIfTI geometry. Ground
truth metrics are used solely inside nested cross-validation to estimate
whether a router generalizes; no case name, image intensity, or cohort field is
available to the model. The exported final linear model can later be applied to
hidden masks without labels if and only if the outer predictions pass the
paired robustness gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage, stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "log_anchor_mm3",
    "log_candidate_mm3",
    "log_volume_ratio",
    "mask_dice",
    "addition_union_fraction",
    "deletion_union_fraction",
    "log_anchor_components",
    "log_candidate_components",
    "component_count_delta",
    "anchor_largest_fraction",
    "candidate_largest_fraction",
)
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02)
CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--anchor-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["Name"]: {
            "dsc": float(row["Lesion_DSC"]),
            "nsd": float(row["Lesion_NSD"]),
        }
        for row in rows
    }


def case_name(path: Path) -> str:
    return path.name.removesuffix(".nii.gz")


def component_summary(mask: np.ndarray) -> tuple[int, float]:
    labels, count = ndimage.label(mask, CONNECTIVITY)
    if count == 0:
        return 0, 0.0
    sizes = np.bincount(labels.reshape(-1))[1:]
    return int(count), float(sizes.max() / max(1, sizes.sum()))


def mask_features(anchor_path: Path, candidate_path: Path) -> list[float]:
    anchor_image = nib.load(anchor_path)
    candidate_image = nib.load(candidate_path)
    if anchor_image.shape != candidate_image.shape or not np.allclose(
        anchor_image.affine, candidate_image.affine, atol=1e-4
    ):
        raise ValueError(f"geometry mismatch: {anchor_path.name}")
    anchor = np.asanyarray(anchor_image.dataobj) > 0
    candidate = np.asanyarray(candidate_image.dataobj) > 0
    voxel_mm3 = float(np.prod(anchor_image.header.get_zooms()[:3]))
    anchor_voxels = int(anchor.sum())
    candidate_voxels = int(candidate.sum())
    intersection = int((anchor & candidate).sum())
    union = int((anchor | candidate).sum())
    additions = int((candidate & ~anchor).sum())
    deletions = int((anchor & ~candidate).sum())
    anchor_components, anchor_largest = component_summary(anchor)
    candidate_components, candidate_largest = component_summary(candidate)
    del anchor, candidate
    return [
        float(np.log1p(anchor_voxels * voxel_mm3)),
        float(np.log1p(candidate_voxels * voxel_mm3)),
        float(np.log((candidate_voxels + 1) / (anchor_voxels + 1))),
        float(2 * intersection / max(1, anchor_voxels + candidate_voxels)),
        float(additions / max(1, union)),
        float(deletions / max(1, union)),
        float(np.log1p(anchor_components)),
        float(np.log1p(candidate_components)),
        float(candidate_components - anchor_components),
        anchor_largest,
        candidate_largest,
    ]


def predict_leave_one_out(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    predictions = np.zeros(len(y), dtype=float)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], y[train])
        predictions[held_out] = float(model.predict(x[[held_out]])[0])
    return predictions


def select_hyperparameters(
    x: np.ndarray, y: np.ndarray, anchor_combined: np.ndarray
) -> tuple[float, float, list[dict[str, float]]]:
    table = []
    for alpha in ALPHAS:
        prediction = predict_leave_one_out(x, y, alpha)
        for threshold in THRESHOLDS:
            choose = prediction > threshold
            mean = float(np.mean(anchor_combined + choose * y))
            table.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "mean_combined": mean,
                    "switches": int(choose.sum()),
                }
            )
    # Prefer validation score, then fewer switches, then a more conservative
    # threshold, then stronger regularization.
    best = max(
        table,
        key=lambda row: (
            row["mean_combined"],
            -row["switches"],
            row["threshold"],
            row["alpha"],
        ),
    )
    return float(best["alpha"]), float(best["threshold"]), table


def bootstrap_ci(values: np.ndarray, seed: int = 20260822) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def main() -> None:
    args = parse_args()
    anchor_paths = {case_name(path): path for path in args.anchor.glob("*.nii.gz")}
    candidate_paths = {
        case_name(path): path for path in args.candidate.glob("*.nii.gz")
    }
    anchor_metrics = load_metrics(args.anchor_metrics)
    candidate_metrics = load_metrics(args.candidate_metrics)
    names = sorted(
        set(anchor_paths)
        & set(candidate_paths)
        & set(anchor_metrics)
        & set(candidate_metrics)
    )
    if len(names) != 50:
        raise ValueError(f"expected 50 aligned public cases, found {len(names)}")

    x = np.asarray(
        [mask_features(anchor_paths[name], candidate_paths[name]) for name in names],
        dtype=float,
    )
    anchor_dsc = np.asarray([anchor_metrics[name]["dsc"] for name in names])
    anchor_nsd = np.asarray([anchor_metrics[name]["nsd"] for name in names])
    candidate_dsc = np.asarray([candidate_metrics[name]["dsc"] for name in names])
    candidate_nsd = np.asarray([candidate_metrics[name]["nsd"] for name in names])
    anchor_combined = (anchor_dsc + anchor_nsd) / 2
    candidate_combined = (candidate_dsc + candidate_nsd) / 2
    target = candidate_combined - anchor_combined

    outer_prediction = np.zeros(len(names), dtype=float)
    outer_alpha = np.zeros(len(names), dtype=float)
    outer_threshold = np.zeros(len(names), dtype=float)
    for held_out in range(len(names)):
        train = np.arange(len(names)) != held_out
        alpha, threshold, _ = select_hyperparameters(
            x[train], target[train], anchor_combined[train]
        )
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], target[train])
        outer_prediction[held_out] = float(model.predict(x[[held_out]])[0])
        outer_alpha[held_out] = alpha
        outer_threshold[held_out] = threshold

    choose = outer_prediction > outer_threshold
    routed_dsc = np.where(choose, candidate_dsc, anchor_dsc)
    routed_nsd = np.where(choose, candidate_nsd, anchor_nsd)
    routed_combined = (routed_dsc + routed_nsd) / 2
    routed_gain = routed_combined - anchor_combined
    oracle_choose = target > 0

    final_alpha, final_threshold, final_table = select_hyperparameters(
        x, target, anchor_combined
    )
    final_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=final_alpha))
    final_pipeline.fit(x, target)
    scaler = final_pipeline.named_steps["standardscaler"]
    ridge = final_pipeline.named_steps["ridge"]

    correlations = {}
    for index, feature in enumerate(FEATURES):
        result = stats.spearmanr(x[:, index], target)
        correlations[feature] = {
            "rho": float(result.statistic),
            "p": float(result.pvalue),
        }

    report = {
        "cases": len(names),
        "features": list(FEATURES),
        "nested_loocv": {
            "switches": int(choose.sum()),
            "dsc": float(routed_dsc.mean()),
            "nsd": float(routed_nsd.mean()),
            "combined": float(routed_combined.mean()),
            "paired_gain": {
                "dsc": float((routed_dsc - anchor_dsc).mean()),
                "nsd": float((routed_nsd - anchor_nsd).mean()),
                "combined": float(routed_gain.mean()),
            },
            "combined_gain_95ci": bootstrap_ci(routed_gain),
            "win_rate": float(np.mean(routed_gain > 1e-8)),
            "tie_rate": float(np.mean(np.abs(routed_gain) <= 1e-8)),
            "catastrophic_regression_rate": float(np.mean(routed_gain < -0.20)),
        },
        "always_anchor": {
            "dsc": float(anchor_dsc.mean()),
            "nsd": float(anchor_nsd.mean()),
            "combined": float(anchor_combined.mean()),
        },
        "always_candidate": {
            "dsc": float(candidate_dsc.mean()),
            "nsd": float(candidate_nsd.mean()),
            "combined": float(candidate_combined.mean()),
        },
        "oracle": {
            "switches": int(oracle_choose.sum()),
            "dsc": float(np.where(oracle_choose, candidate_dsc, anchor_dsc).mean()),
            "nsd": float(np.where(oracle_choose, candidate_nsd, anchor_nsd).mean()),
            "combined": float(np.maximum(anchor_combined, candidate_combined).mean()),
        },
        "final_model": {
            "alpha": final_alpha,
            "threshold": final_threshold,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "ridge_intercept": float(ridge.intercept_),
            "ridge_coefficients": ridge.coef_.tolist(),
            "inner_grid": final_table,
        },
        "feature_target_spearman": correlations,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "outer_predictions.csv").open("w", newline="") as handle:
        fieldnames = [
            "case",
            "choose_candidate",
            "predicted_combined_gain",
            "threshold",
            "alpha",
            "actual_combined_gain",
            *FEATURES,
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "case": name,
                    "choose_candidate": int(choose[index]),
                    "predicted_combined_gain": outer_prediction[index],
                    "threshold": outer_threshold[index],
                    "alpha": outer_alpha[index],
                    "actual_combined_gain": target[index],
                    **dict(zip(FEATURES, x[index])),
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
