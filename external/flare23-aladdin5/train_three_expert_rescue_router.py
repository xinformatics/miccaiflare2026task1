#!/usr/bin/env python3
"""Nested-OOF FLARE23 rescue router above the frozen GF/FLARE25 route."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONNECTIVITY = np.ones((3, 3, 3), dtype=np.uint8)
CONFIDENCE_FIELDS = (
    "tumor_probability_max",
    "tumor_probability_mean_positive",
    "tumor_probability_q10_positive",
    "tumor_probability_q50_positive",
    "tumor_margin_mean_positive",
    "tumor_margin_q10_positive",
    "tumor_fraction_crop",
)
EXPERTS = ("gf", "f25", "f23")
PAIRS = (("gf", "f25"), ("gf", "f23"), ("f25", "f23"))
GEOMETRY_FEATURES = (
    "log_voxel_mm3", "log_dim0", "log_dim1", "log_dim2",
    "log_extent0_mm", "log_extent1_mm", "log_extent2_mm",
)
MASK_FEATURE_SUFFIXES = (
    "log_mm3", "log_components", "log_largest_mm3", "largest_fraction",
    "centroid0", "centroid1", "centroid2", "span0", "span1", "span2",
)
PAIR_FEATURE_SUFFIXES = (
    "dice", "jaccard", "a_only_fraction", "b_only_fraction",
    "a_intersection_fraction", "b_intersection_fraction",
)
FEATURES = (
    *GEOMETRY_FEATURES,
    *(f"{expert}_{suffix}" for expert in EXPERTS for suffix in MASK_FEATURE_SUFFIXES),
    *(f"{a}_{b}_{suffix}" for a, b in PAIRS for suffix in PAIR_FEATURE_SUFFIXES),
    "stage1_choose_f25", "stage1_predicted_gain",
    *(f"f23_{field}" for field in CONFIDENCE_FIELDS),
    "f23_roi_foreground_fraction", "f23_crop_volume_fraction",
    "f23_crop_span0", "f23_crop_span1", "f23_crop_span2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_confidence(paths: list[Path]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"incomplete confidence terminal: {path}")
        for record in payload["cases"]:
            name = record["case"].removesuffix(".nii.gz")
            if name in result:
                raise RuntimeError(f"duplicate confidence case: {name}")
            result[name] = record
    return result


def mask_summary(mask: np.ndarray, voxel_mm3: float) -> tuple[float, ...]:
    voxels = int(mask.sum())
    if voxels == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    points = np.argwhere(mask)
    lower = points.min(axis=0); upper = points.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    labels, count = ndimage.label(mask[crop], structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max())
    centroid = points.mean(axis=0) / np.maximum(1, np.asarray(mask.shape) - 1)
    span = (upper - lower) / np.asarray(mask.shape)
    return (
        float(voxels * voxel_mm3), float(count), float(largest * voxel_mm3),
        float(largest / voxels), *map(float, centroid), *map(float, span),
    )


def pair_summary(a: np.ndarray, b: np.ndarray) -> tuple[float, ...]:
    av = int(a.sum()); bv = int(b.sum())
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return (
        2.0 * intersection / max(1, av + bv),
        intersection / max(1, union),
        (av - intersection) / max(1, union),
        (bv - intersection) / max(1, union),
        intersection / max(1, av),
        intersection / max(1, bv),
    )


def extract_one(task: tuple[dict, str, str, str, dict, dict]) -> tuple[str, list[float]]:
    item, gf_dir, f25_dir, f23_dir, stage1, confidence = task
    filename = item["prediction_name"]
    images = {
        "gf": nib.load(str(Path(gf_dir) / filename)),
        "f25": nib.load(str(Path(f25_dir) / filename)),
        "f23": nib.load(str(Path(f23_dir) / filename)),
    }
    reference = images["gf"]
    for expert in ("f25", "f23"):
        if images[expert].shape != reference.shape or not np.allclose(
            images[expert].affine, reference.affine, rtol=0, atol=1e-4
        ):
            raise ValueError(f"geometry mismatch for {filename}: {expert}")
    masks = {key: np.asanyarray(image.dataobj) > 0 for key, image in images.items()}
    voxel_mm3 = float(abs(np.linalg.det(reference.affine[:3, :3])))
    dims = np.asarray(reference.shape, dtype=float)
    spacing = np.asarray(reference.header.get_zooms()[:3], dtype=float)
    values: list[float] = [
        float(np.log1p(voxel_mm3)),
        *map(float, np.log1p(dims)),
        *map(float, np.log1p(dims * spacing)),
    ]
    for expert in EXPERTS:
        volume, count, largest, largest_fraction, *position = mask_summary(
            masks[expert], voxel_mm3
        )
        values.extend(
            [float(np.log1p(volume)), float(np.log1p(count)),
             float(np.log1p(largest)), largest_fraction, *position]
        )
    for a, b in PAIRS:
        values.extend(pair_summary(masks[a], masks[b]))
    values.extend(
        [float(stage1["choose_flare"]), float(stage1["predicted_gain"])]
    )
    confidence_values = confidence["confidence"]
    values.extend(float(confidence_values[field]) for field in CONFIDENCE_FIELDS)
    roi_shape = np.asarray(confidence["roi_shape_zyx"], dtype=float)
    roi_fraction = float(confidence["roi_foreground_voxels"] / max(1.0, roi_shape.prod()))
    crop = confidence.get("crop_zyx")
    if crop is None:
        crop_spans = np.zeros(3, dtype=float)
    else:
        crop_spans = np.asarray([hi - lo for lo, hi in crop], dtype=float)
        crop_spans /= np.asarray(confidence["shape_zyx"], dtype=float)
    values.extend([roi_fraction, float(crop_spans.prod()), *map(float, crop_spans)])
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError(f"invalid feature vector for {filename}: {len(values)}")
    return item["name"], values


def candidates() -> list[dict]:
    result = [{"family": "ridge", "alpha": value} for value in (0.1, 1.0, 10.0, 100.0)]
    result.extend(
        {"family": "hist_gradient_boosting", "max_leaf_nodes": leaves,
         "l2_regularization": l2}
        for leaves in (5, 7, 15) for l2 in (1.0, 10.0)
    )
    return result


def make_model(spec: dict):
    if spec["family"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    return HistGradientBoostingRegressor(
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        l2_regularization=float(spec["l2_regularization"]),
        learning_rate=0.05, max_iter=150, min_samples_leaf=20,
        random_state=20260829,
    )


THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15)


def select_inner(x: np.ndarray, target: np.ndarray, base: np.ndarray,
                 strata: np.ndarray) -> tuple[dict, float]:
    cv = StratifiedKFold(4, shuffle=True, random_state=20260829)
    table = []
    for spec in candidates():
        predicted = np.zeros(len(target), dtype=float)
        for train, valid in cv.split(x, strata):
            model = make_model(spec); model.fit(x[train], target[train])
            predicted[valid] = model.predict(x[valid])
        for threshold in THRESHOLDS:
            choose = predicted > threshold
            routed = base + choose * target
            table.append((float(routed.mean()), -int(choose.sum()), threshold, spec))
    _, _, threshold, spec = max(table, key=lambda row: (row[0], row[1], row[2]))
    return spec, float(threshold)


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260829)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), (0.025, 0.975))]


def read_csv_by(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row[key]: row for row in csv.DictReader(stream)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--f25", required=True, type=Path)
    parser.add_argument("--f23", required=True, type=Path)
    parser.add_argument("--pair-cases", required=True, type=Path)
    parser.add_argument("--f23-cases", required=True, type=Path)
    parser.add_argument("--stage1-oof", required=True, type=Path)
    parser.add_argument("--confidence-terminal", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("cases") != 707 or len(manifest.get("items", [])) != 707:
        raise RuntimeError("manifest is not exact-707")
    pair_scores = read_csv_by(args.pair_cases, "case")
    f23_scores = read_csv_by(args.f23_cases, "case")
    stage1 = read_csv_by(args.stage1_oof, "case")
    confidence = load_confidence(args.confidence_terminal)
    names = [item["name"] for item in manifest["items"]]
    if any(set(mapping) != set(names) for mapping in (pair_scores, f23_scores, stage1, confidence)):
        raise RuntimeError("training inputs do not share the exact-707 case inventory")
    tasks = [
        (item, str(args.gf), str(args.f25), str(args.f23), stage1[item["name"]],
         confidence[item["name"]]) for item in manifest["items"]
    ]
    extracted: dict[str, list[float]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, values) in enumerate(pool.map(extract_one, tasks, chunksize=1), 1):
            extracted[name] = values
            if index % 25 == 0 or index == 707:
                print(f"three-expert features {index}/707", flush=True)
    x = np.asarray([extracted[name] for name in names], dtype=float)
    gf = np.asarray([float(pair_scores[name]["gf_matched_dice"]) for name in names])
    f25 = np.asarray([float(pair_scores[name]["flare25_matched_dice"]) for name in names])
    f23 = np.asarray([float(f23_scores[name]["flare25_matched_dice"]) for name in names])
    base_choose = np.asarray([int(stage1[name]["choose_flare"]) > 0 for name in names])
    base = np.where(base_choose, f25, gf)
    target = f23 - base
    cohorts = np.asarray([item["cohort"] for item in manifest["items"]])
    outer = StratifiedKFold(5, shuffle=True, random_state=20260829)
    predicted = np.zeros(707); thresholds = np.zeros(707); folds = np.zeros(707, dtype=int)
    selected = []
    for fold, (train, valid) in enumerate(outer.split(x, cohorts)):
        spec, threshold = select_inner(x[train], target[train], base[train], cohorts[train])
        model = make_model(spec); model.fit(x[train], target[train])
        predicted[valid] = model.predict(x[valid]); thresholds[valid] = threshold
        folds[valid] = fold; selected.append({"fold": fold, "spec": spec, "threshold": threshold})
    choose = predicted > thresholds
    routed = np.where(choose, f23, base)
    rescue_gain = routed - base; total_gain = routed - gf
    fold_rescue = [float(rescue_gain[folds == fold].mean()) for fold in range(5)]
    rescue_ci = bootstrap_ci(rescue_gain)
    final_spec, final_threshold = select_inner(x, target, base, cohorts)
    final_model = make_model(final_spec); final_model.fit(x, target)
    args.output.mkdir(parents=True, exist_ok=True)
    model_payload = {"features": list(FEATURES), "spec": final_spec,
                     "threshold": final_threshold}
    if final_spec["family"] == "ridge":
        scaler = final_model.named_steps["standardscaler"]
        ridge = final_model.named_steps["ridge"]
        model_payload.update({"scaler_mean": scaler.mean_.tolist(),
                              "scaler_scale": scaler.scale_.tolist(),
                              "intercept": float(ridge.intercept_),
                              "coefficients": ridge.coef_.tolist()})
    else:
        model_file = args.output / "rescue_router.pkl"
        model_file.write_bytes(pickle.dumps(final_model, protocol=5))
        model_payload["pickle"] = {"path": str(model_file.resolve()),
                                   "sha256": sha256(model_file)}
    with (args.output / "features_and_oof.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["case", "cohort", "fold", "gf_score", "stage1_score", "f23_score",
                  "actual_rescue_gain", "predicted_rescue_gain", "threshold",
                  "choose_f23", *FEATURES]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow({"case": name, "cohort": cohorts[i], "fold": folds[i],
                             "gf_score": gf[i], "stage1_score": base[i], "f23_score": f23[i],
                             "actual_rescue_gain": target[i],
                             "predicted_rescue_gain": predicted[i], "threshold": thresholds[i],
                             "choose_f23": int(choose[i]), **dict(zip(FEATURES, x[i]))})
    checks = {
        "overall_gain_over_gf_ge_0p015": float(total_gain.mean()) >= 0.015,
        "rescue_ci_lower_above_zero": rescue_ci[0] > 0,
        "at_least_four_positive_rescue_folds": sum(value > 0 for value in fold_rescue) >= 4,
        "catastrophic_regression_rate_le_0p10": float(np.mean(rescue_gain < -0.20)) <= 0.10,
    }
    report = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE", "cases": 707,
        "stage1_oof_mean": float(base.mean()),
        "stage1_oof_gain_over_gf": float((base - gf).mean()),
        "three_expert_oof_mean": float(routed.mean()),
        "three_expert_oof_gain_over_gf": float(total_gain.mean()),
        "rescue": {"incremental_gain": float(rescue_gain.mean()),
                   "gain_95ci": rescue_ci, "switches": int(choose.sum()),
                   "switch_precision": float(np.mean(target[choose] > 0)) if choose.any() else 0.0,
                   "fold_gains": fold_rescue,
                   "catastrophic_regression_rate": float(np.mean(rescue_gain < -0.20))},
        "gate_checks": checks, "router_audit_gate_pass": all(checks.values()),
        "selected_outer": selected, "final_model": model_payload,
        "assets": {"manifest_sha256": sha256(args.manifest),
                   "pair_cases_sha256": sha256(args.pair_cases),
                   "f23_cases_sha256": sha256(args.f23_cases),
                   "stage1_oof_sha256": sha256(args.stage1_oof),
                   "confidence_terminal_sha256": [sha256(path) for path in args.confidence_terminal]},
        "constraints": {"case_name_feature": False, "cohort_feature": False,
                        "ground_truth_feature": False, "public_read": False,
                        "hidden_read": False, "submission_created": False, "uploaded": False},
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
