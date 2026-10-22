#!/usr/bin/env python3
"""Train one grouped fold of the candidate-specific 3-D refiner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from candidate_refiner_model import CandidateRefiner, candidate_refiner_loss
from candidate_refiner_sampling import (
    hierarchical_sample_weights,
    sampling_audit,
    semantic_source_group,
)
from candidate_refiner_source_conditioning import (
    DEFAULT_SOURCE_CONDITION_GROUPS,
    append_source_condition_channels,
    source_condition_index,
)
from focus_validation_metrics import aggregate_positive_dice
from fault_tolerant_wandb import (
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
)


def stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def seed_worker(worker_id: int) -> None:
    """Seed every RNG used inside a persistent data-loader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_reproducibility(seed: int) -> None:
    """Configure seeded CPU/CUDA execution for repeatable controlled runs."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def grouped_fold_rows(
    rows: list[dict[str, str]], folds: int, fold: int, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if folds < 2 or not 0 <= fold < folds:
        raise ValueError("invalid grouped fold")
    always_train = [
        row for row in rows if int(str(row.get("training_only", "0"))) != 0
    ]
    selectable = [
        row for row in rows if int(str(row.get("training_only", "0"))) == 0
    ]
    if not selectable:
        raise ValueError("no fold-selectable rows")
    by_cohort: dict[str, set[str]] = {}
    for row in selectable:
        by_cohort.setdefault(row["cohort"], set()).add(row["case"])
    validation_cases: set[str] = set()
    for cohort, cases in sorted(by_cohort.items()):
        ordered = sorted(cases, key=lambda case: stable_digest(f"{cohort}:{case}", seed))
        validation_cases.update(ordered[fold::folds])
    train = always_train + [
        row for row in selectable if row["case"] not in validation_cases
    ]
    validation = [
        row for row in selectable if row["case"] in validation_cases
    ]
    if not train or not validation:
        raise ValueError("empty grouped split")
    if {row["case"] for row in train} & {row["case"] for row in validation}:
        raise AssertionError("case leakage across grouped split")
    if any(int(str(row.get("training_only", "0"))) != 0 for row in validation):
        raise AssertionError("training-only row leaked into grouped validation")
    return train, validation


def filter_rows_by_source_groups(
    rows: list[dict[str, str]], source_groups: list[str] | tuple[str, ...]
) -> list[dict[str, str]]:
    """Select semantic proposal families without changing held-out membership."""

    requested = tuple(sorted({str(value).strip() for value in source_groups}))
    if not requested or any(not value for value in requested):
        raise ValueError("at least one nonempty source group is required")
    observed = {
        semantic_source_group(row.get("source_name", "")) for row in rows
    }
    missing = set(requested) - observed
    if missing:
        raise ValueError(f"source groups absent from rows: {sorted(missing)}")
    selected = [
        row
        for row in rows
        if semantic_source_group(row.get("source_name", "")) in requested
    ]
    if not selected:
        raise ValueError("source-group filter selected no rows")
    if {int(row["label"]) for row in selected} != {0, 1}:
        raise ValueError("source-group training subset must contain both labels")
    return selected


class PatchDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        augment: bool,
        source_condition_groups: tuple[str, ...] = (),
    ):
        self.rows = rows
        self.augment = augment
        self.source_condition_groups = tuple(source_condition_groups)
        if self.source_condition_groups:
            for row in rows:
                source_condition_index(
                    row.get("source_name", ""), self.source_condition_groups
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=False) as archive:
            image = torch.from_numpy(archive["image"].astype(np.float32))
            target = torch.from_numpy(archive["target"].astype(np.float32))
            label = torch.tensor(float(archive["label"]), dtype=torch.float32)
        if self.augment:
            for axis in (1, 2, 3):
                if torch.rand(()) < 0.5:
                    image = torch.flip(image, dims=(axis,))
                    target = torch.flip(target, dims=(axis,))
            if torch.rand(()) < 0.3:
                scale = 0.9 + 0.2 * torch.rand(())
                shift = -0.05 + 0.1 * torch.rand(())
                image[:2] = torch.clamp(image[:2] * scale + shift, 0.0, 1.0)
        if self.source_condition_groups:
            image = append_source_condition_channels(
                image, row.get("source_name", ""), self.source_condition_groups
            )
        return image, target, label, row["patch_id"]


def threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> dict[str, float]:
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if not (len(labels) == len(probabilities) == len(weights)):
        raise ValueError("classification arrays do not align")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("classification weights must be finite and positive")
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        selected = probabilities >= threshold
        tp = float(weights[selected & (labels == 1)].sum())
        fp = float(weights[selected & (labels == 0)].sum())
        fn = float(weights[~selected & (labels == 1)].sum())
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        beta2 = 0.25
        fbeta = (1 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
        candidate = (fbeta, precision, recall, -threshold)
        if best is None or candidate > best[0]:
            best = (candidate, threshold, precision, recall, fbeta)
    assert best is not None
    return {
        "threshold": float(best[1]),
        "precision": float(best[2]),
        "recall": float(best[3]),
        "f0p5": float(best[4]),
    }


def apply_case_balanced_validation_metrics(
    metrics: dict[str, float],
    predictions: dict[str, np.ndarray],
    validation_rows: list[dict[str, str]],
) -> dict[str, float | int | str]:
    """Align per-epoch checkpoint selection with case-averaged evaluation."""

    metadata = {str(row["patch_id"]): row for row in validation_rows}
    patch_ids = [str(value) for value in predictions["patch_id"]]
    if len(metadata) != len(validation_rows) or len(patch_ids) != len(set(patch_ids)):
        raise ValueError("validation patch IDs are not unique")
    if set(patch_ids) != set(metadata):
        raise ValueError("validation classification coverage mismatch")
    case_counts: dict[str, int] = {}
    cases = []
    for patch_id in patch_ids:
        case = str(metadata[patch_id].get("case", ""))
        if not case:
            raise ValueError(f"missing validation case identity: {patch_id}")
        cases.append(case)
        case_counts[case] = case_counts.get(case, 0) + 1
    weights = np.asarray([1.0 / case_counts[case] for case in cases])
    labels = np.asarray(predictions["label"], dtype=np.uint8)
    probabilities = np.asarray(
        predictions["classification_probability"], dtype=np.float64
    )
    classification = threshold_metrics(labels, probabilities, weights)
    segmentation = aggregate_positive_dice(
        predictions["positive_patch_id"],
        predictions["segmentation_dice_grid"],
        predictions["segmentation_thresholds"],
        validation_rows,
    )
    updated: dict[str, float | int | str] = dict(metrics)
    updated.update(
        {
            "patch_pooled_classification_threshold": float(metrics["threshold"]),
            "patch_pooled_classification_f0p5": float(metrics["f0p5"]),
            "patch_pooled_average_precision": float(metrics["average_precision"]),
            "patch_pooled_roc_auc": float(metrics["roc_auc"]),
            "patch_mean_segmentation_dice": float(metrics["segmentation_dice"]),
            "aggregation": "patches weighted to equal total weight per case",
            "cases": len(case_counts),
            **classification,
            "average_precision": float(
                average_precision_score(labels, probabilities, sample_weight=weights)
            ),
            "roc_auc": (
                float(roc_auc_score(labels, probabilities, sample_weight=weights))
                if np.unique(labels).size == 2
                else float("nan")
            ),
            "segmentation_threshold": float(segmentation["threshold"]),
            "segmentation_dice": float(segmentation["case_balanced_dice"]),
            "segmentation_dice_at_0p5": float(
                segmentation["case_balanced_dice_at_0p5"]
            ),
            "case_balanced_segmentation_dice": float(
                segmentation["case_balanced_dice"]
            ),
            "cohort_balanced_segmentation_dice": float(
                segmentation["cohort_balanced_dice"]
            ),
            "positive_cases": int(segmentation["positive_cases"]),
            "positive_cohorts": int(segmentation["positive_cohorts"]),
        }
    )
    updated["selection_score"] = 0.5 * float(updated["average_precision"]) + 0.5 * float(
        updated["segmentation_dice"]
    )
    return updated


def apply_source_group_segmentation_metrics(
    metrics: dict[str, float | int | str],
    predictions: dict[str, np.ndarray],
    validation_rows: list[dict[str, str]],
    source_groups: list[str] | tuple[str, ...],
) -> dict[str, float | int | str | list[str]]:
    """Add a case-balanced contour selector for routed proposal families.

    Validation remains exhaustive. Only the positive Dice rows belonging to
    the declared semantic source families participate in this additional
    checkpoint-selection metric.
    """

    requested = tuple(sorted({str(value).strip() for value in source_groups}))
    if not requested or any(not value for value in requested):
        raise ValueError("routed segmentation source groups must be nonempty")
    selected_rows = [
        row
        for row in validation_rows
        if semantic_source_group(row.get("source_name", "")) in requested
    ]
    observed = {
        semantic_source_group(row.get("source_name", "")) for row in selected_rows
    }
    missing = set(requested) - observed
    if missing:
        raise ValueError(
            f"routed segmentation groups absent from validation: {sorted(missing)}"
        )
    selected_positive_ids = {
        str(row["patch_id"]) for row in selected_rows if int(row["label"]) == 1
    }
    all_positive_ids = np.asarray(predictions["positive_patch_id"]).astype(str)
    keep = np.asarray(
        [patch_id in selected_positive_ids for patch_id in all_positive_ids],
        dtype=bool,
    )
    if int(keep.sum()) != len(selected_positive_ids):
        raise ValueError("routed segmentation positive coverage mismatch")
    segmentation = aggregate_positive_dice(
        all_positive_ids[keep],
        np.asarray(predictions["segmentation_dice_grid"])[keep],
        predictions["segmentation_thresholds"],
        selected_rows,
    )
    updated: dict[str, float | int | str | list[str]] = dict(metrics)
    updated.update(
        {
            "routed_segmentation_source_groups": list(requested),
            "routed_segmentation_threshold": float(segmentation["threshold"]),
            "routed_segmentation_dice": float(
                segmentation["case_balanced_dice"]
            ),
            "routed_cohort_balanced_segmentation_dice": float(
                segmentation["cohort_balanced_dice"]
            ),
            "routed_segmentation_positive_cases": int(
                segmentation["positive_cases"]
            ),
            "routed_segmentation_positive_cohorts": int(
                segmentation["positive_cohorts"]
            ),
        }
    )
    return updated


@torch.no_grad()
def evaluate_with_predictions(
    model: CandidateRefiner,
    loader: DataLoader,
    device: torch.device,
    validation_rows: list[dict[str, str]] | None = None,
    segmentation_threshold_max: float = 0.80,
) -> tuple[dict[str, float | int | str], dict[str, np.ndarray]]:
    model.eval()
    patch_ids, labels, probabilities = [], [], []
    positive_patch_ids: list[str] = []
    segmentation_dice_rows: list[np.ndarray] = []
    threshold_steps = round((float(segmentation_threshold_max) - 0.20) / 0.05)
    if (
        not np.isfinite(segmentation_threshold_max)
        or not 0.50 <= segmentation_threshold_max <= 0.95
        or abs(0.20 + 0.05 * threshold_steps - segmentation_threshold_max) > 1e-8
    ):
        raise ValueError(
            "segmentation_threshold_max must be a 0.05 grid point in [0.50, 0.95]"
        )
    segmentation_thresholds = np.linspace(
        0.20, float(segmentation_threshold_max), threshold_steps + 1
    )
    segmentation_dice_sums = np.zeros(len(segmentation_thresholds), dtype=np.float64)
    segmentation_positive_count = 0
    for images, targets, batch_labels, batch_patch_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        class_logits, seg_logits = model(images)
        patch_ids.extend(map(str, batch_patch_ids))
        labels.extend(batch_labels.numpy().astype(np.uint8).tolist())
        probabilities.extend(torch.sigmoid(class_logits).cpu().tolist())
        positive = batch_labels > 0.5
        if bool(positive.any()):
            segmentation_probabilities = torch.sigmoid(seg_logits[positive.to(device)])
            truth = targets[positive.to(device)] > 0.5
            threshold_dice = []
            for threshold_index, threshold in enumerate(segmentation_thresholds):
                prediction = segmentation_probabilities >= float(threshold)
                intersection = (prediction & truth).sum(dim=(1, 2, 3, 4)).float()
                denominator = prediction.sum(dim=(1, 2, 3, 4)) + truth.sum(
                    dim=(1, 2, 3, 4)
                )
                dice = (2 * intersection + 1) / (denominator + 1)
                segmentation_dice_sums[threshold_index] += float(dice.sum().cpu())
                threshold_dice.append(dice.cpu().numpy())
            positive_indices = np.flatnonzero(positive.numpy())
            positive_patch_ids.extend(str(batch_patch_ids[index]) for index in positive_indices)
            segmentation_dice_rows.append(np.stack(threshold_dice, axis=1))
            segmentation_positive_count += int(positive.sum())
    labels_array = np.asarray(labels, dtype=np.uint8)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    result = threshold_metrics(labels_array, probabilities_array)
    unique_labels = np.unique(labels_array)
    segmentation_dice_values = segmentation_dice_sums / max(
        segmentation_positive_count, 1
    )
    # Prefer a threshold closer to 0.50 when validation Dice ties. This makes
    # the calibrated contour cutoff deterministic and conservative.
    segmentation_index = max(
        range(len(segmentation_thresholds)),
        key=lambda index: (
            float(segmentation_dice_values[index]),
            -abs(float(segmentation_thresholds[index]) - 0.50),
        ),
    )
    threshold_0p5_index = int(np.argmin(np.abs(segmentation_thresholds - 0.50)))
    result.update(
        {
            "average_precision": float(average_precision_score(labels_array, probabilities_array)),
            "roc_auc": (
                float(roc_auc_score(labels_array, probabilities_array))
                if unique_labels.size == 2
                else float("nan")
            ),
            "segmentation_threshold": float(segmentation_thresholds[segmentation_index]),
            "segmentation_dice": float(segmentation_dice_values[segmentation_index]),
            "segmentation_dice_at_0p5": float(
                segmentation_dice_values[threshold_0p5_index]
            ),
            "positive": int(labels_array.sum()),
            "negative": int((labels_array == 0).sum()),
        }
    )
    result["selection_score"] = 0.5 * result["average_precision"] + 0.5 * result["segmentation_dice"]
    details = {
        "patch_id": np.asarray(patch_ids),
        "label": labels_array,
        "classification_probability": probabilities_array.astype(np.float32),
        "positive_patch_id": np.asarray(positive_patch_ids),
        "segmentation_thresholds": segmentation_thresholds.astype(np.float32),
        "segmentation_dice_grid": (
            np.concatenate(segmentation_dice_rows, axis=0).astype(np.float32)
            if segmentation_dice_rows
            else np.empty((0, len(segmentation_thresholds)), dtype=np.float32)
        ),
    }
    if validation_rows is not None:
        result = apply_case_balanced_validation_metrics(
            result, details, validation_rows
        )
    return result, details


def evaluate(
    model: CandidateRefiner,
    loader: DataLoader,
    device: torch.device,
    segmentation_threshold_max: float = 0.80,
) -> dict[str, float]:
    return evaluate_with_predictions(
        model,
        loader,
        device,
        segmentation_threshold_max=segmentation_threshold_max,
    )[0]


def atomic_save_predictions(path: Path, details: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **details)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def learning_rate(step: int, total_steps: int, warmup_steps: int, peak: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def epoch_sample_count(row_count: int, batch_size: int, requested: int) -> int:
    """Resolve a bounded epoch while preserving the historical zero default."""

    if row_count < 1 or batch_size < 1 or requested < 0:
        raise ValueError("invalid epoch sample-count request")
    return requested if requested > 0 else max(row_count, batch_size * 100)


def validation_batch_size(training_batch_size: int, requested: int) -> int:
    """Resolve a separately bounded inference-only validation batch size."""

    if training_batch_size < 1 or requested < 0:
        raise ValueError("invalid validation batch-size request")
    return requested if requested > 0 else training_batch_size


def should_validate_epoch(epoch: int, epochs: int, interval: int) -> bool:
    """Validate a fixed sparse trajectory while always covering endpoints."""

    if epochs < 1 or interval < 1 or not 1 <= epoch <= epochs:
        raise ValueError("invalid validation schedule")
    return epoch == 1 or epoch == epochs or (epoch - 1) % interval == 0


def validate_resume_configuration(
    saved: dict[str, object], current: argparse.Namespace
) -> None:
    """Reject recipe drift while accepting checkpoints made before new defaults."""

    legacy_defaults = {
        "samples_per_epoch": 0,
        "train_source_groups": [],
        "segmentation_selection_source_groups": [],
        "balance_source_groups": False,
        "source_conditioning": False,
        "source_condition_groups": [],
        "prompt_masked_classifier": False,
        "prompt_channel": 2,
        "small_lesion_loss": False,
        "small_lesion_positive_weight_cap": 100.0,
        "classification_negative_weight": 1.0,
        "negative_segmentation_weight": 0.1,
        "segmentation_threshold_max": 0.80,
        "validation_interval": 1,
        "validation_batch_size": 0,
    }
    keys = (
        "folds",
        "epochs",
        "batch_size",
        "workers",
        "lr",
        "weight_decay",
        "base_channels",
        "sampling_temperature",
        "balance_source_groups",
        "samples_per_epoch",
        "train_source_groups",
        "segmentation_selection_source_groups",
        "source_conditioning",
        "source_condition_groups",
        "prompt_masked_classifier",
        "prompt_channel",
        "small_lesion_loss",
        "small_lesion_positive_weight_cap",
        "classification_negative_weight",
        "negative_segmentation_weight",
        "segmentation_threshold_max",
        "validation_interval",
        "validation_batch_size",
        "seed",
    )
    for key in keys:
        saved_value = saved.get(key, legacy_defaults.get(key))
        current_value = getattr(current, key, legacy_defaults.get(key))
        if saved_value != current_value:
            raise ValueError(f"cannot resume with changed training option: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=0.0,
        help=(
            "0 preserves label-balanced v2 sampling; 1 fully balances "
            "label/cohort/case cells; intermediate values geometrically interpolate"
        ),
    )
    parser.add_argument(
        "--balance-source-groups",
        action="store_true",
        help=(
            "within each label/cohort/case cell, balance semantic proposal "
            "families; lung_cell0..7 count as one lung_cells family"
        ),
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=0,
        help=(
            "bounded weighted draws per epoch; zero preserves the historical "
            "full training-row count"
        ),
    )
    parser.add_argument(
        "--train-source-group",
        dest="train_source_groups",
        action="append",
        default=[],
        help=(
            "repeatable semantic proposal family used for optimizer batches; "
            "validation remains exhaustive across the complete held-out fold"
        ),
    )
    parser.add_argument(
        "--segmentation-selection-source-group",
        dest="segmentation_selection_source_groups",
        action="append",
        default=[],
        help=(
            "repeatable semantic proposal family used only to select "
            "checkpoint_best_routed_segmentation.pt"
        ),
    )
    parser.add_argument(
        "--source-conditioning",
        action="store_true",
        help=(
            "append fixed one-hot channels for epoch105, currentgf, weak AutoPET, "
            "weak LNDb, or pooled lung-cell proposal provenance"
        ),
    )
    parser.add_argument(
        "--prompt-masked-classifier",
        action="store_true",
        help=(
            "augment global candidate-presence pooling with average/max encoder "
            "features restricted to the fixed candidate-prompt channel"
        ),
    )
    parser.add_argument(
        "--small-lesion-loss",
        action="store_true",
        help=(
            "use per-positive-patch foreground weighting so larger lesions in "
            "a batch do not suppress the BCE signal from tiny lesions"
        ),
    )
    parser.add_argument(
        "--small-lesion-positive-weight-cap",
        type=float,
        default=100.0,
        help="maximum per-patch positive BCE weight for --small-lesion-loss",
    )
    parser.add_argument(
        "--classification-negative-weight",
        type=float,
        default=1.0,
        help=(
            "relative BCE weight for annotation-negative candidate patches. "
            "Values below one treat negatives as uncertain/unlabeled while "
            "normalizing the effective classification-loss scale"
        ),
    )
    parser.add_argument(
        "--negative-segmentation-weight",
        type=float,
        default=0.1,
        help=(
            "relative softplus penalty on segmentation logits for annotation-negative "
            "patches; lower values are an opt-in positive-unlabeled treatment"
        ),
    )
    parser.add_argument(
        "--segmentation-threshold-max",
        type=float,
        default=0.80,
        help=(
            "upper endpoint of the 0.05 validation contour grid; 0.80 "
            "preserves the historical evaluator"
        ),
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help=(
            "run exhaustive held-out validation every N epochs, always at "
            "epochs 1 and final; optimizer checkpoints are still saved each epoch"
        ),
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=0,
        help=(
            "inference-only held-out batch size; zero reuses --batch-size. "
            "This does not alter optimizer updates or the learning-rate schedule"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--wandb-project", default="pancancer-ct")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.train_source_groups = sorted(set(args.train_source_groups))
    args.segmentation_selection_source_groups = sorted(
        set(args.segmentation_selection_source_groups)
    )
    args.source_condition_groups = (
        list(DEFAULT_SOURCE_CONDITION_GROUPS) if args.source_conditioning else []
    )
    # Candidate patch layout is a fixed interface shared by preparation and
    # deployment: (soft, lung, seed_support). Persist the semantic channel in
    # every new checkpoint so prompted inference cannot silently drift.
    args.prompt_channel = 2
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.lr <= 0
        or args.small_lesion_positive_weight_cap < 1
        or not 0 < args.classification_negative_weight <= 1
        or not 0 <= args.negative_segmentation_weight <= 1
        or not 0.50 <= args.segmentation_threshold_max <= 0.95
        or abs(
            0.20
            + 0.05 * round((args.segmentation_threshold_max - 0.20) / 0.05)
            - args.segmentation_threshold_max
        )
        > 1e-8
        or args.samples_per_epoch < 0
        or args.validation_interval < 1
        or args.validation_batch_size < 0
        or not 0 <= args.sampling_temperature <= 1
    ):
        parser.error("invalid training configuration")
    configure_reproducibility(args.seed + args.fold)
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    all_train_rows, validation_rows = grouped_fold_rows(
        rows, args.folds, args.fold, args.seed
    )
    train_rows = (
        filter_rows_by_source_groups(all_train_rows, args.train_source_groups)
        if args.train_source_groups
        else all_train_rows
    )
    if args.segmentation_selection_source_groups:
        # Validate group spelling and held-out coverage before allocating GPU
        # memory. This call does not replace exhaustive validation rows.
        filter_rows_by_source_groups(
            validation_rows, args.segmentation_selection_source_groups
        )
    sample_weights = hierarchical_sample_weights(
        train_rows, args.sampling_temperature, args.balance_source_groups
    )
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader_generator = torch.Generator().manual_seed(
        args.seed + args.fold + 10_000
    )
    validation_loader_generator = torch.Generator().manual_seed(
        args.seed + args.fold + 20_000
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights),
        num_samples=epoch_sample_count(
            len(train_rows), args.batch_size, args.samples_per_epoch
        ),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        PatchDataset(
            train_rows,
            augment=True,
            source_condition_groups=tuple(args.source_condition_groups),
        ),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )
    validation_loader = DataLoader(
        PatchDataset(
            validation_rows,
            augment=False,
            source_condition_groups=tuple(args.source_condition_groups),
        ),
        batch_size=validation_batch_size(
            args.batch_size, args.validation_batch_size
        ),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
        generator=validation_loader_generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateRefiner(
        in_channels=3 + len(args.source_condition_groups),
        base_channels=args.base_channels,
        prompt_masked_classifier=args.prompt_masked_classifier,
        prompt_channel=args.prompt_channel,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(5 * len(train_loader), total_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    index_digest = hashlib.sha256(args.index.read_bytes()).hexdigest()
    sampler_report = sampling_audit(
        train_rows, args.sampling_temperature, args.balance_source_groups
    )
    sampler_report.update(
        {
            "fold": args.fold,
            "folds": args.folds,
            "seed": args.seed,
            "index": str(args.index.resolve()),
            "index_sha256": index_digest,
            "samples_per_epoch": epoch_sample_count(
                len(train_rows), args.batch_size, args.samples_per_epoch
            ),
            "unfiltered_training_patches": len(all_train_rows),
            "training_source_groups": list(args.train_source_groups),
            "validation_patches": len(validation_rows),
            "validation_source_groups": sorted(
                {
                    semantic_source_group(row.get("source_name", ""))
                    for row in validation_rows
                }
            ),
            "segmentation_selection_source_groups": list(
                args.segmentation_selection_source_groups
            ),
        }
    )
    (args.output / "sampling_audit.json").write_text(
        json.dumps(sampler_report, indent=2) + "\n",
        encoding="utf-8",
    )
    best_score = -1.0
    best_segmentation = -1.0
    best_cohort_segmentation = -1.0
    best_classification = -1.0
    best_routed_segmentation = -1.0
    start_epoch = 1
    global_step = 0
    history = []
    checkpoint_metrics: dict[str, object] = {}
    last_path = args.output / "checkpoint_last.pt"
    history_path = args.output / "history.json"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("index_sha256") != index_digest:
            raise ValueError("cannot resume with a different patch index")
        if int(checkpoint.get("fold", -1)) != args.fold:
            raise ValueError("cannot resume a different grouped fold")
        validate_resume_configuration(checkpoint.get("config", {}), args)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", (start_epoch - 1) * len(train_loader))
        )
        best_score = float(checkpoint.get("best_score", -1.0))
        best_segmentation = float(checkpoint.get("best_segmentation", -1.0))
        best_cohort_segmentation = float(
            checkpoint.get(
                "best_cohort_segmentation",
                checkpoint.get("metrics", {}).get(
                    "cohort_balanced_segmentation_dice", -1.0
                ),
            )
        )
        best_classification = float(checkpoint.get("best_classification", -1.0))
        best_routed_segmentation = float(
            checkpoint.get("best_routed_segmentation", -1.0)
        )
        checkpoint_metrics = dict(checkpoint.get("metrics", {}))
        generator_states = checkpoint.get("generator_states", {})
        if generator_states:
            generator.set_state(generator_states["sampler"])
            train_loader_generator.set_state(generator_states["train_loader"])
            validation_loader_generator.set_state(
                generator_states["validation_loader"]
            )
        if isinstance(checkpoint.get("history"), list):
            history = checkpoint["history"]
        elif history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(
            f"resuming fold {args.fold} at epoch {start_epoch}; best={best_score:.6f}",
            flush=True,
        )
    run = None
    if args.wandb_name:
        wandb_run_id = hashlib.sha256(
            f"{args.output.resolve()}::{args.fold}".encode("utf-8")
        ).hexdigest()[:16]
        run = safe_wandb_init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=wandb_run_id,
            resume="allow",
            dir=str(args.output),
            config=vars(args),
        )
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for images, targets, labels, _ in train_loader:
            lr = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                class_logits, segmentation_logits = model(images)
                loss, parts = candidate_refiner_loss(
                    class_logits,
                    segmentation_logits,
                    labels,
                    targets,
                    classification_negative_weight=(
                        args.classification_negative_weight
                    ),
                    negative_segmentation_weight=(
                        args.negative_segmentation_weight
                    ),
                    small_lesion_loss=args.small_lesion_loss,
                    small_lesion_positive_weight_cap=(
                        args.small_lesion_positive_weight_cap
                    ),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            global_step += 1
        training_seconds = time.time() - started
        validation_due = should_validate_epoch(
            epoch, args.epochs, args.validation_interval
        )
        validation_predictions = None
        metrics = None
        validation_seconds = 0.0
        if validation_due:
            validation_started = time.time()
            metrics, validation_predictions = evaluate_with_predictions(
                model,
                validation_loader,
                device,
                validation_rows,
                segmentation_threshold_max=args.segmentation_threshold_max,
            )
            if args.segmentation_selection_source_groups:
                metrics = apply_source_group_segmentation_metrics(
                    metrics,
                    validation_predictions,
                    validation_rows,
                    args.segmentation_selection_source_groups,
                )
            validation_seconds = time.time() - validation_started
            checkpoint_metrics = dict(metrics)
        trace_record = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "validated": validation_due,
        }
        if metrics is not None:
            trace_record.update(metrics)
            history.append(trace_record)
        print(json.dumps(trace_record, sort_keys=True), flush=True)
        if run is not None:
            run = safe_wandb_log(run, trace_record, step=epoch)
        is_best = metrics is not None and metrics["selection_score"] > best_score
        is_best_segmentation = (
            metrics is not None and metrics["segmentation_dice"] > best_segmentation
        )
        is_best_cohort_segmentation = (
            metrics is not None
            and metrics["cohort_balanced_segmentation_dice"]
            > best_cohort_segmentation
        )
        is_best_classification = (
            metrics is not None and metrics["average_precision"] > best_classification
        )
        is_best_routed_segmentation = (
            metrics is not None
            and args.segmentation_selection_source_groups
            and metrics["routed_segmentation_dice"] > best_routed_segmentation
        )
        if is_best:
            best_score = metrics["selection_score"]
        if is_best_segmentation:
            best_segmentation = metrics["segmentation_dice"]
        if is_best_cohort_segmentation:
            best_cohort_segmentation = metrics[
                "cohort_balanced_segmentation_dice"
            ]
        if is_best_classification:
            best_classification = metrics["average_precision"]
        if is_best_routed_segmentation:
            best_routed_segmentation = metrics["routed_segmentation_dice"]
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "fold": args.fold,
            "folds": args.folds,
            "config": vars(args),
            "metrics": checkpoint_metrics,
            "best_score": best_score,
            "best_segmentation": best_segmentation,
            "best_cohort_segmentation": best_cohort_segmentation,
            "best_classification": best_classification,
            "best_routed_segmentation": best_routed_segmentation,
            "index_sha256": index_digest,
            "global_step": global_step,
            "history": history,
            "generator_states": {
                "sampler": generator.get_state(),
                "train_loader": train_loader_generator.get_state(),
                "validation_loader": validation_loader_generator.get_state(),
            },
        }
        torch.save(checkpoint, args.output / "checkpoint_last.pt")
        if metrics is not None:
            # Retain every scored point, not only whichever aggregate selector
            # happened to improve. Source-routed policies can then choose a
            # mature/discovery optimum reproducibly after the trajectory ends.
            torch.save(
                checkpoint,
                args.output / f"checkpoint_epoch_{epoch:04d}.pt",
            )
            assert validation_predictions is not None
            atomic_save_predictions(
                args.output / f"validation_predictions_epoch_{epoch:04d}.npz",
                validation_predictions,
            )
        if is_best:
            assert validation_predictions is not None
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
            atomic_save_predictions(
                args.output / "validation_predictions_best.npz",
                validation_predictions,
            )
        if is_best_segmentation:
            assert validation_predictions is not None
            torch.save(
                checkpoint, args.output / "checkpoint_best_segmentation.pt"
            )
            atomic_save_predictions(
                args.output / "validation_predictions_best_segmentation.npz",
                validation_predictions,
            )
        if is_best_cohort_segmentation:
            assert validation_predictions is not None
            torch.save(
                checkpoint,
                args.output / "checkpoint_best_cohort_segmentation.pt",
            )
            atomic_save_predictions(
                args.output / "validation_predictions_best_cohort_segmentation.npz",
                validation_predictions,
            )
        if is_best_classification:
            assert validation_predictions is not None
            torch.save(
                checkpoint, args.output / "checkpoint_best_classification.pt"
            )
            atomic_save_predictions(
                args.output / "validation_predictions_best_classification.npz",
                validation_predictions,
            )
        if is_best_routed_segmentation:
            assert validation_predictions is not None
            torch.save(
                checkpoint,
                args.output / "checkpoint_best_routed_segmentation.pt",
            )
            atomic_save_predictions(
                args.output / "validation_predictions_best_routed_segmentation.npz",
                validation_predictions,
            )
        history_path.write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
    history_path.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "COMPLETE.txt").write_text(
        __import__("datetime").datetime.now().astimezone().isoformat() + "\n"
    )
    safe_wandb_finish(run)


if __name__ == "__main__":
    main()
