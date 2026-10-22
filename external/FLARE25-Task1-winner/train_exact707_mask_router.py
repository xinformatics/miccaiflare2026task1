#!/usr/bin/env python3
"""Nested-CV, inference-visible router for GF versus official FLARE25 masks."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
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
FEATURES = (
    "log_voxel_mm3", "log_dim0", "log_dim1", "log_dim2",
    "log_extent0_mm", "log_extent1_mm", "log_extent2_mm",
    "log_gf_mm3", "log_flare_mm3", "log_volume_ratio",
    "mask_dice", "mask_jaccard", "flare_addition_fraction",
    "flare_deletion_fraction", "gf_intersection_fraction",
    "flare_intersection_fraction", "log_gf_components",
    "log_flare_components", "component_count_delta",
    "log_gf_largest_mm3", "log_flare_largest_mm3",
    "gf_largest_fraction", "flare_largest_fraction",
    "gf_centroid0", "gf_centroid1", "gf_centroid2",
    "flare_centroid0", "flare_centroid1", "flare_centroid2",
    "gf_span0", "gf_span1", "gf_span2",
    "flare_span0", "flare_span1", "flare_span2",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mask_summary(mask: np.ndarray, voxel_mm3: float) -> tuple[float, ...]:
    count_voxels = int(mask.sum())
    if count_voxels == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    labels, count = ndimage.label(mask, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel())[1:]
    largest = int(sizes.max())
    points = np.argwhere(mask)
    centroid = points.mean(axis=0) / np.maximum(1, np.asarray(mask.shape) - 1)
    span = (points.max(axis=0) - points.min(axis=0) + 1) / np.asarray(mask.shape)
    return (
        float(count_voxels * voxel_mm3), float(count), float(largest * voxel_mm3),
        float(largest / count_voxels), *map(float, centroid), *map(float, span),
    )


def extract_one(task: tuple[dict, str, str]) -> tuple[str, list[float]]:
    item, gf_dir, flare_dir = task
    name = item["prediction_name"]
    gf_image = nib.load(str(Path(gf_dir) / name))
    flare_image = nib.load(str(Path(flare_dir) / name))
    if gf_image.shape != flare_image.shape or not np.allclose(
        gf_image.affine, flare_image.affine, rtol=0, atol=1e-4
    ):
        raise ValueError(f"geometry mismatch: {name}")
    gf = np.asanyarray(gf_image.dataobj) > 0
    flare = np.asanyarray(flare_image.dataobj) > 0
    voxel_mm3 = float(abs(np.linalg.det(gf_image.affine[:3, :3])))
    gf_stats = mask_summary(gf, voxel_mm3)
    flare_stats = mask_summary(flare, voxel_mm3)
    gv, gc, gl, glf, *gpos = gf_stats
    fv, fc, fl, flf, *fpos = flare_stats
    intersection = int(np.count_nonzero(gf & flare))
    union = int(np.count_nonzero(gf | flare))
    gvox = int(gf.sum()); fvox = int(flare.sum())
    dims = np.asarray(gf.shape, dtype=float)
    spacing = np.asarray(gf_image.header.get_zooms()[:3], dtype=float)
    values = [
        np.log1p(voxel_mm3), *np.log1p(dims), *np.log1p(dims * spacing),
        np.log1p(gv), np.log1p(fv), np.log((fv + 1.0) / (gv + 1.0)),
        2.0 * intersection / max(1, gvox + fvox), intersection / max(1, union),
        (fvox - intersection) / max(1, union),
        (gvox - intersection) / max(1, union),
        intersection / max(1, gvox), intersection / max(1, fvox),
        np.log1p(gc), np.log1p(fc), fc - gc,
        np.log1p(gl), np.log1p(fl), glf, flf, *gpos, *fpos,
    ]
    if len(values) != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError(f"invalid features for {name}: {len(values)}")
    return item["name"], [float(value) for value in values]


def make_model(spec: dict):
    if spec["family"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(spec["alpha"])))
    return HistGradientBoostingRegressor(
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        l2_regularization=float(spec["l2_regularization"]),
        learning_rate=0.05, max_iter=150, min_samples_leaf=20,
        random_state=20260828,
    )


def candidates(policy: dict) -> list[dict]:
    grid = policy["model_grid"]
    result = [{"family": "ridge", "alpha": value} for value in grid["ridge_alpha"]]
    result += [
        {"family": "hist_gradient_boosting", "max_leaf_nodes": leaves,
         "l2_regularization": l2}
        for leaves in grid["hgb_max_leaf_nodes"]
        for l2 in grid["hgb_l2_regularization"]
    ]
    return result


def split_labels(cohorts: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray([f"{cohort}|{int(gain > 0)}" for cohort, gain in zip(cohorts, target)])


def select_inner(x: np.ndarray, y: np.ndarray, gf: np.ndarray, strata: np.ndarray,
                 policy: dict) -> tuple[dict, float, list[dict]]:
    folds = int(policy["validation"]["inner_folds"])
    cv = StratifiedKFold(folds, shuffle=True, random_state=20260828)
    table = []
    for spec in candidates(policy):
        prediction = np.zeros(len(y), dtype=float)
        for train, valid in cv.split(x, strata):
            model = make_model(spec); model.fit(x[train], y[train])
            prediction[valid] = model.predict(x[valid])
        for threshold in policy["model_grid"]["switch_threshold"]:
            choose = prediction > float(threshold)
            routed = gf + choose * y
            table.append({
                "spec": spec, "threshold": float(threshold),
                "routed_mean": float(routed.mean()),
                "gain": float((routed - gf).mean()), "switches": int(choose.sum()),
            })
    best = max(table, key=lambda row: (row["routed_mean"], -row["switches"], row["threshold"]))
    return best["spec"], best["threshold"], table


def bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(20260828)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return [float(x) for x in np.quantile(values[indices].mean(axis=1), (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--gf", required=True, type=Path)
    parser.add_argument("--flare", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("cases") != 707 or sha256(args.manifest) != policy["training_set"]["manifest_sha256"]:
        raise RuntimeError("frozen exact-707 manifest mismatch")
    if not all(item.get("complete_label") is True for item in manifest["items"]):
        raise RuntimeError("manifest contains a non-complete training label")
    with args.cases.open(newline="") as stream:
        metric_rows = {row["case"]: row for row in csv.DictReader(stream)}
    if len(metric_rows) != 707:
        raise RuntimeError("metric inventory is not 707")
    tasks = [(item, str(args.gf), str(args.flare)) for item in manifest["items"]]
    extracted = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (name, values) in enumerate(pool.map(extract_one, tasks, chunksize=1), 1):
            extracted[name] = values
            if index % 25 == 0 or index == 707:
                print(f"features {index}/707", flush=True)
    names = [item["name"] for item in manifest["items"]]
    x = np.asarray([extracted[name] for name in names], dtype=float)
    gf = np.asarray([float(metric_rows[name]["gf_matched_dice"]) for name in names])
    flare = np.asarray([float(metric_rows[name]["flare25_matched_dice"]) for name in names])
    target = flare - gf
    cohorts = np.asarray([item["cohort"] for item in manifest["items"]])
    strata = split_labels(cohorts, target)
    outer = StratifiedKFold(int(policy["validation"]["outer_folds"]), shuffle=True,
                            random_state=int(policy["validation"]["random_seed"]))
    prediction = np.zeros(707, dtype=float); threshold = np.zeros(707, dtype=float)
    folds = np.zeros(707, dtype=int); selected = []
    for fold, (train, valid) in enumerate(outer.split(x, strata)):
        spec, cut, _ = select_inner(x[train], target[train], gf[train], strata[train], policy)
        model = make_model(spec); model.fit(x[train], target[train])
        prediction[valid] = model.predict(x[valid]); threshold[valid] = cut; folds[valid] = fold
        selected.append({"fold": fold, "spec": spec, "threshold": cut})
    choose = prediction > threshold
    routed = np.where(choose, flare, gf)
    gain = routed - gf
    fold_gains = [float(gain[folds == fold].mean()) for fold in range(5)]
    ci = bootstrap_ci(gain)
    validation = policy["validation"]
    checks = {
        "minimum_oof_gain": float(gain.mean()) >= float(validation["minimum_oof_matched_dice_gain"]),
        "bootstrap_ci_lower_above_zero": ci[0] > 0,
        "positive_outer_folds": sum(value > 0 for value in fold_gains) >= int(validation["minimum_positive_outer_folds"]),
        "catastrophic_regression_rate": float(np.mean(gain < float(validation["catastrophic_regression_delta"]))) <= float(validation["maximum_catastrophic_regression_rate"]),
    }
    final_spec, final_threshold, inner_table = select_inner(x, target, gf, strata, policy)
    final_model = make_model(final_spec); final_model.fit(x, target)
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
        import pickle
        model_file = args.output / "router_model.pkl"
        args.output.mkdir(parents=True, exist_ok=True)
        model_file.write_bytes(pickle.dumps(final_model, protocol=5))
        model_payload["pickle"] = {"path": str(model_file.resolve()), "sha256": sha256(model_file)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "features_and_oof.csv").open("w", newline="") as stream:
        fields = ["case", "cohort", "fold", "gf_matched_dice", "flare_matched_dice",
                  "actual_gain", "predicted_gain", "threshold", "choose_flare", *FEATURES]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow({"case": name, "cohort": cohorts[i], "fold": folds[i],
                             "gf_matched_dice": gf[i], "flare_matched_dice": flare[i],
                             "actual_gain": target[i], "predicted_gain": prediction[i],
                             "threshold": threshold[i], "choose_flare": int(choose[i]),
                             **dict(zip(FEATURES, x[i]))})
    report = {
        "schema_version": 1, "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE", "cases": 707,
        "always_gf_mean_matched_dice": float(gf.mean()),
        "always_flare_mean_matched_dice": float(flare.mean()),
        "oof": {"routed_mean_matched_dice": float(routed.mean()),
                "gain_over_gf": float(gain.mean()), "gain_95ci": ci,
                "switches": int(choose.sum()),
                "switch_precision": float(np.mean(target[choose] > 0)) if choose.any() else 0.0,
                "fold_gains": fold_gains,
                "catastrophic_regression_rate": float(np.mean(gain < -0.20))},
        "gate_checks": checks, "router_audit_gate_pass": all(checks.values()),
        "selected_outer": selected, "final_model": model_payload,
        "assets": {"manifest_sha256": sha256(args.manifest), "cases_sha256": sha256(args.cases),
                   "policy_sha256": sha256(args.policy)},
        "constraints": {"case_name_feature": False, "cohort_feature": False,
                        "ground_truth_feature": False, "public_read": False,
                        "hidden_read": False, "submission_created": False, "uploaded": False}
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
