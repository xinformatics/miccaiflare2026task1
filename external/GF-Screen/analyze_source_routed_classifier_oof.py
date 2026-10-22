#!/usr/bin/env python3
"""Audit source-family checkpoint routing on exact grouped held-out patches."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from analyze_focus_classifier_fusion_oof import metrics
from focus_classifier_fusion import SOURCE_GROUP_NAMES, source_group
from train_candidate_refiner import grouped_fold_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (
            [str(value) for value in archive["patch_id"]],
            archive["label"].astype(np.uint8),
            archive["classification_probability"].astype(np.float64),
        )


def choose(audits: list[tuple[Path, dict]], group: str) -> tuple[Path, dict]:
    candidates = []
    for path, payload in audits:
        report = payload.get("source_classification_groups", {}).get(group)
        if not isinstance(report, dict):
            raise ValueError(f"source classification group absent: {path} {group}")
        candidates.append(
            (
                float(report["average_precision"]),
                float(report["f0p5"]),
                -abs(float(report["threshold"]) - 0.5),
                -int(payload["epoch"]),
                path,
                payload,
            )
        )
    _, _, _, _, path, payload = max(candidates, key=lambda value: value[:4])
    return path, payload


def source_nonregression(
    baseline: dict[str, object],
    candidate: dict[str, object],
    maximum_drop: float = 0.005,
) -> bool:
    """Protect both ranking quality and the calibrated operating point."""

    if maximum_drop < 0:
        raise ValueError("maximum source-metric drop must be nonnegative")
    return all(
        float(candidate[name]) >= float(baseline[name]) - maximum_drop
        for name in ("average_precision", "f0p5")
    )


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.folds < 2 or len(args.run) != args.folds or args.bootstrap < 0:
        parser.error("one run directory per fold and a nonnegative bootstrap are required")

    with args.index.open(newline="", encoding="utf-8") as stream:
        index_rows = list(csv.DictReader(stream))
    by_id = {str(row["patch_id"]): row for row in index_rows}
    if not index_rows or len(by_id) != len(index_rows):
        raise ValueError("invalid or duplicate exact707 patch index")
    index_digest = sha256(args.index)
    all_ids: list[str] = []
    all_labels: list[int] = []
    all_baseline: list[float] = []
    all_candidate: list[float] = []
    all_folds: list[int] = []
    selections = []
    for fold, run in enumerate(args.run):
        _, held_out = grouped_fold_rows(index_rows, args.folds, fold, args.seed)
        expected_ids = {str(row["patch_id"]) for row in held_out}
        audit_paths = sorted(args.audit_root.glob(f"fold{fold}_source_epoch*.json"))
        audits: list[tuple[Path, dict]] = []
        for path in audit_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = Path(payload["checkpoint"])
            prediction = Path(payload["prediction"])
            if (
                int(payload.get("fold", -1)) != fold
                or int(payload.get("folds", -1)) != args.folds
                or int(payload.get("seed", -1)) != args.seed
                or payload.get("index_sha256") != index_digest
                or sha256(checkpoint) != payload.get("checkpoint_sha256")
                or sha256(prediction) != payload.get("prediction_sha256")
            ):
                raise ValueError(f"invalid source-audit provenance: {path}")
            audits.append((path, payload))
        if not audits:
            raise ValueError(f"no scored source audits for fold {fold}")

        baseline_path = run / "validation_predictions_best_classification.npz"
        baseline_checkpoint = run / "checkpoint_best_classification.pt"
        baseline_ids, baseline_labels, baseline_probabilities = load_prediction(
            baseline_path
        )
        if len(baseline_ids) != len(set(baseline_ids)) or set(baseline_ids) != expected_ids:
            raise ValueError(f"baseline held-out coverage mismatch in fold {fold}")
        baseline_position = {
            patch_id: index for index, patch_id in enumerate(baseline_ids)
        }
        route_archives = {}
        fold_selections = {}
        for group in SOURCE_GROUP_NAMES:
            audit_path, selected = choose(audits, group)
            prediction = Path(selected["prediction"])
            checkpoint = Path(selected["checkpoint"])
            ids, labels, probabilities = load_prediction(prediction)
            if len(ids) != len(set(ids)) or set(ids) != expected_ids:
                raise ValueError(f"routed held-out coverage mismatch: fold={fold} group={group}")
            route_archives[group] = (
                {patch_id: index for index, patch_id in enumerate(ids)},
                labels,
                probabilities,
            )
            report = selected["source_classification_groups"][group]
            fold_selections[group] = {
                "epoch": int(selected["epoch"]),
                "average_precision": float(report["average_precision"]),
                "f0p5": float(report["f0p5"]),
                "threshold": float(report["threshold"]),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction": str(prediction.resolve()),
                "prediction_sha256": sha256(prediction),
                "audit": str(audit_path.resolve()),
                "audit_sha256": sha256(audit_path),
            }
        for patch_id in sorted(expected_ids):
            row = by_id[patch_id]
            expected_label = int(row["label"])
            baseline_index = baseline_position[patch_id]
            group = source_group(str(row["source_name"]))
            position, labels, probabilities = route_archives[group]
            route_index = position[patch_id]
            if (
                int(baseline_labels[baseline_index]) != expected_label
                or int(labels[route_index]) != expected_label
            ):
                raise ValueError(f"classification label mismatch: {patch_id}")
            all_ids.append(patch_id)
            all_labels.append(expected_label)
            all_baseline.append(float(baseline_probabilities[baseline_index]))
            all_candidate.append(float(probabilities[route_index]))
            all_folds.append(fold)
        selections.append(
            {
                "fold": fold,
                "baseline_checkpoint": str(baseline_checkpoint.resolve()),
                "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
                "routes": fold_selections,
            }
        )
    if len(all_ids) != len(index_rows) or set(all_ids) != set(by_id):
        raise ValueError("source-routed classifier lacks exact index coverage")

    rows = [by_id[patch_id] for patch_id in all_ids]
    labels = np.asarray(all_labels, dtype=np.uint8)
    baseline = np.asarray(all_baseline, dtype=np.float64)
    candidate = np.asarray(all_candidate, dtype=np.float64)
    baseline_metrics = metrics(rows, labels, baseline)
    candidate_metrics = metrics(rows, labels, candidate)
    groups = np.asarray([source_group(str(row["source_name"])) for row in rows])
    by_source = {}
    source_checks = []
    for group in SOURCE_GROUP_NAMES:
        selected = np.flatnonzero(groups == group)
        source_rows = [rows[int(index)] for index in selected]
        before = metrics(source_rows, labels[selected], baseline[selected])
        after = metrics(source_rows, labels[selected], candidate[selected])
        gain = float(after["average_precision"]) - float(before["average_precision"])
        f_gain = float(after["f0p5"]) - float(before["f0p5"])
        by_source[group] = {
            "patches": len(selected),
            "baseline": before,
            "candidate": after,
            "average_precision_gain": gain,
            "f0p5_gain": f_gain,
        }
        source_checks.append(source_nonregression(before, after))

    rng = np.random.default_rng(args.seed)
    case_names = sorted({str(row["case"]) for row in rows})
    case_indices = {
        case: np.asarray(
            [index for index, row in enumerate(rows) if str(row["case"]) == case],
            dtype=np.int64,
        )
        for case in case_names
    }
    bootstrap_gains = []
    for _ in range(args.bootstrap):
        sampled = rng.choice(case_names, size=len(case_names), replace=True)
        indices = np.concatenate([case_indices[case] for case in sampled])
        weights = np.concatenate(
            [np.full(len(case_indices[case]), 1.0 / len(case_indices[case])) for case in sampled]
        )
        if np.unique(labels[indices]).size < 2:
            continue
        bootstrap_gains.append(
            float(
                average_precision_score(labels[indices], candidate[indices], sample_weight=weights)
                - average_precision_score(labels[indices], baseline[indices], sample_weight=weights)
            )
        )
    ap_gain = float(candidate_metrics["average_precision"]) - float(
        baseline_metrics["average_precision"]
    )
    f_gain = float(candidate_metrics["f0p5"]) - float(baseline_metrics["f0p5"])
    checks = {
        "average_precision_gain": ap_gain >= 0.005,
        "f0p5_gain": f_gain >= 0.005,
        "source_nonregression": all(source_checks),
    }
    decision = (
        "QUALIFY_SOURCE_ROUTED_CLASSIFIER"
        if all(checks.values())
        else "KEEP_GLOBAL_CLASSIFIER"
    )
    atomic_npz(
        args.predictions_output,
        patch_id=np.asarray(all_ids),
        label=labels,
        baseline_probability=baseline.astype(np.float32),
        classification_probability=candidate.astype(np.float32),
        fold=np.asarray(all_folds, dtype=np.uint8),
    )
    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "selection_basis": "source-family checkpoint routing on exact707 grouped held-out patches",
        "index": str(args.index.resolve()),
        "index_sha256": index_digest,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "average_precision_gain": ap_gain,
        "f0p5_gain": f_gain,
        "average_precision_gain_95ci": (
            [float(value) for value in np.quantile(bootstrap_gains, (0.025, 0.975))]
            if bootstrap_gains
            else None
        ),
        "by_source": by_source,
        "selections": selections,
        "checks": checks,
        "decision": decision,
        "predictions": str(args.predictions_output.resolve()),
        "predictions_sha256": sha256(args.predictions_output),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
