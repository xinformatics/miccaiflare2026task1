#!/usr/bin/env python3
"""Nested cross-validation for an image-only router between two segmenters.

The router uses scan geometry and a coarse, label-free HU histogram. Ground
truth metrics are used only as the nested-CV training target. Case identifiers,
filenames, masks, and cohort metadata are excluded from the feature vector.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "log_voxels",
    "extent_x_mm",
    "extent_y_mm",
    "extent_z_mm",
    "spacing_z_mm",
    "air_fraction",
    "body_fraction",
    "body_hu_p10",
    "body_hu_p50",
    "body_hu_p90",
    "soft_tissue_mean",
    "soft_tissue_std",
    "high_hu_fraction",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--anchor-metrics", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["Name"]: {
            "dsc": float(row["Lesion_DSC"]),
            "nsd": float(row["Lesion_NSD"]),
        }
        for row in rows
    }


def image_case(path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not path.name.endswith(suffix):
        raise ValueError(f"unexpected image name: {path.name}")
    return path.name[: -len(suffix)]


def image_features(path: Path) -> list[float]:
    image = nib.load(str(path))
    shape = np.asarray(image.shape[:3], dtype=float)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    # A deterministic coarse lattice keeps peak memory low and avoids learning
    # scanner-specific interpolation artifacts from a random sample.
    coarse = np.asarray(image.dataobj[::8, ::8, ::4], dtype=np.float32)
    coarse = np.clip(coarse[np.isfinite(coarse)], -1024.0, 2000.0)
    if coarse.size == 0:
        raise ValueError(f"no finite CT voxels: {path}")
    body = coarse[coarse > -500.0]
    soft = body[(body >= -200.0) & (body <= 300.0)]
    if body.size < 100 or soft.size < 100:
        raise ValueError(f"insufficient body voxels in coarse sample: {path}")
    p10, p50, p90 = np.percentile(body, (10, 50, 90))
    return [
        float(np.log1p(np.prod(shape))),
        float(shape[0] * spacing[0]),
        float(shape[1] * spacing[1]),
        float(shape[2] * spacing[2]),
        float(spacing[2]),
        float(np.mean(coarse <= -500.0)),
        float(np.mean(coarse > -500.0)),
        float(p10),
        float(p50),
        float(p90),
        float(np.mean(soft)),
        float(np.std(soft)),
        float(np.mean(body > 300.0)),
    ]


def loocv_prediction(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    prediction = np.zeros(len(y), dtype=float)
    for held_out in range(len(y)):
        train = np.arange(len(y)) != held_out
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], y[train])
        prediction[held_out] = float(model.predict(x[[held_out]])[0])
    return prediction


def select_hyperparameters(
    x: np.ndarray, target: np.ndarray, anchor_combined: np.ndarray
) -> tuple[float, float, list[dict]]:
    table = []
    for alpha in ALPHAS:
        prediction = loocv_prediction(x, target, alpha)
        for threshold in THRESHOLDS:
            choose = prediction > threshold
            table.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "mean_combined": float(np.mean(anchor_combined + choose * target)),
                    "switches": int(choose.sum()),
                }
            )
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
    return [float(value) for value in np.quantile(values[indices].mean(1), (0.025, 0.975))]


def main() -> None:
    args = parse_args()
    image_paths = {image_case(path): path for path in args.images.glob("*_0000.nii.gz")}
    anchor = load_metrics(args.anchor_metrics)
    candidate = load_metrics(args.candidate_metrics)
    names = sorted(set(image_paths) & set(anchor) & set(candidate))
    if len(names) != 50:
        raise ValueError(f"expected 50 aligned public cases, found {len(names)}")
    x = np.asarray([image_features(image_paths[name]) for name in names], dtype=float)
    if x.shape != (50, len(FEATURES)) or not np.all(np.isfinite(x)):
        raise ValueError(f"invalid feature matrix: {x.shape}")

    anchor_dsc = np.asarray([anchor[name]["dsc"] for name in names])
    anchor_nsd = np.asarray([anchor[name]["nsd"] for name in names])
    candidate_dsc = np.asarray([candidate[name]["dsc"] for name in names])
    candidate_nsd = np.asarray([candidate[name]["nsd"] for name in names])
    anchor_combined = (anchor_dsc + anchor_nsd) / 2
    candidate_combined = (candidate_dsc + candidate_nsd) / 2
    target = candidate_combined - anchor_combined

    outer_prediction = np.zeros(len(names))
    outer_alpha = np.zeros(len(names))
    outer_threshold = np.zeros(len(names))
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
    gain = routed_combined - anchor_combined

    final_alpha, final_threshold, inner_grid = select_hyperparameters(
        x, target, anchor_combined
    )
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=final_alpha))
    final_model.fit(x, target)
    scaler = final_model.named_steps["standardscaler"]
    ridge = final_model.named_steps["ridge"]
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
                "combined": float(gain.mean()),
            },
            "combined_gain_95ci": bootstrap_ci(gain),
            "win_rate": float(np.mean(gain > 1e-8)),
            "tie_rate": float(np.mean(np.abs(gain) <= 1e-8)),
            "catastrophic_regression_rate": float(np.mean(gain < -0.20)),
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
            "switches": int(np.sum(target > 0)),
            "dsc": float(np.where(target > 0, candidate_dsc, anchor_dsc).mean()),
            "nsd": float(np.where(target > 0, candidate_nsd, anchor_nsd).mean()),
            "combined": float(np.maximum(anchor_combined, candidate_combined).mean()),
        },
        "final_model": {
            "alpha": final_alpha,
            "threshold": final_threshold,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "ridge_intercept": float(ridge.intercept_),
            "ridge_coefficients": ridge.coef_.tolist(),
            "inner_grid": inner_grid,
        },
        "feature_target_spearman": correlations,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "features_and_targets.npz",
        names=np.asarray(names),
        features=x,
        target=target,
        outer_prediction=outer_prediction,
        outer_threshold=outer_threshold,
        outer_alpha=outer_alpha,
        choose_candidate=choose,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
