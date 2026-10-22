#!/usr/bin/env python3
"""Evaluate a saved candidate refiner on a different, case-matched patch index.

This is a training-label-only transfer diagnostic. It preserves the checkpoint's
original grouped fold and records both index hashes so a cross-domain result
cannot be mistaken for the checkpoint's native validation metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from candidate_refiner_model import CandidateRefiner
from train_candidate_refiner import (
    PatchDataset,
    atomic_save_predictions,
    configure_reproducibility,
    evaluate_with_predictions,
    grouped_fold_rows,
    seed_worker,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_spec(checkpoint: dict) -> dict[str, int | float | bool]:
    config = checkpoint.get("config", {})
    state = checkpoint.get("state_dict", {})
    input_weight = state.get("encoder0.conv1.weight")
    if not isinstance(input_weight, torch.Tensor) or input_weight.ndim != 5:
        raise ValueError("checkpoint is missing the candidate-refiner input projection")
    observed_base = int(input_weight.shape[0])
    observed_input = int(input_weight.shape[1])
    required = ("fold", "folds", "seed")
    if any(key not in checkpoint and key not in config for key in required):
        raise ValueError("checkpoint is missing grouped-fold identity")
    fold = int(checkpoint.get("fold", config.get("fold")))
    folds = int(checkpoint.get("folds", config.get("folds")))
    seed = int(config.get("seed", checkpoint.get("seed", -1)))
    configured_base = int(config.get("base_channels", observed_base))
    prompt_channel = int(config.get("prompt_channel", 2))
    segmentation_threshold_max = float(config.get("segmentation_threshold_max", 0.80))
    if configured_base != observed_base or observed_input < 1:
        raise ValueError("checkpoint architecture metadata mismatch")
    if prompt_channel != 2:
        raise ValueError(
            "checkpoint prompt channel does not match the fixed "
            "(soft, lung, seed_support) input layout"
        )
    threshold_steps = round((segmentation_threshold_max - 0.20) / 0.05)
    if folds < 2 or not 0 <= fold < folds or seed < 0:
        raise ValueError("invalid grouped-fold identity")
    if (
        not np.isfinite(segmentation_threshold_max)
        or not 0.50 <= segmentation_threshold_max <= 0.95
        or abs(0.20 + 0.05 * threshold_steps - segmentation_threshold_max) > 1e-8
    ):
        raise ValueError("invalid checkpoint segmentation-threshold grid")
    return {
        "fold": fold,
        "folds": folds,
        "seed": seed,
        "base_channels": observed_base,
        "in_channels": observed_input,
        "prompt_masked_classifier": bool(
            config.get("prompt_masked_classifier", False)
        ),
        "prompt_channel": prompt_channel,
        "segmentation_threshold_max": segmentation_threshold_max,
    }


def evaluate_transfer(
    index: Path,
    checkpoint_path: Path,
    output: Path,
    predictions: Path,
    batch_size: int,
    workers: int,
    allow_index_transfer: bool,
    device: torch.device | None = None,
) -> dict:
    if batch_size < 1 or workers < 0:
        raise ValueError("invalid loader configuration")
    index_digest = sha256(index)
    checkpoint_digest = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec = checkpoint_spec(checkpoint)
    training_index_digest = str(checkpoint.get("index_sha256", ""))
    if not training_index_digest:
        raise ValueError("checkpoint is missing its training-index hash")
    is_transfer = training_index_digest != index_digest
    if is_transfer and not allow_index_transfer:
        raise ValueError("evaluation index differs from training index; pass --allow-index-transfer")

    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _, validation_rows = grouped_fold_rows(
        rows, spec["folds"], spec["fold"], spec["seed"]
    )
    configure_reproducibility(spec["seed"] + spec["fold"])
    generator = torch.Generator().manual_seed(spec["seed"] + spec["fold"] + 20_000)
    loader = DataLoader(
        PatchDataset(validation_rows, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    selected_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandidateRefiner(
        in_channels=int(spec["in_channels"]),
        base_channels=int(spec["base_channels"]),
        prompt_masked_classifier=bool(spec["prompt_masked_classifier"]),
        prompt_channel=int(spec["prompt_channel"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics, arrays = evaluate_with_predictions(
        model,
        loader,
        selected_device,
        validation_rows,
        segmentation_threshold_max=float(spec["segmentation_threshold_max"]),
    )
    atomic_save_predictions(predictions, arrays)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "training_index_sha256": training_index_digest,
        "evaluation_index": str(index.resolve()),
        "evaluation_index_sha256": index_digest,
        "cross_index_transfer": is_transfer,
        "fold": spec["fold"],
        "folds": spec["folds"],
        "seed": spec["seed"],
        "architecture": {
            "in_channels": spec["in_channels"],
            "base_channels": spec["base_channels"],
            "prompt_channel": spec["prompt_channel"],
            "prompt_channel_semantic": "seed_support",
        },
        "segmentation_threshold_max": spec["segmentation_threshold_max"],
        "validation_cases": len({row["case"] for row in validation_rows}),
        "validation_patches": len(validation_rows),
        "metrics": metrics,
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256(predictions),
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-index-transfer", action="store_true")
    args = parser.parse_args()
    payload = evaluate_transfer(
        args.index,
        args.checkpoint,
        args.output,
        args.predictions,
        args.batch_size,
        args.workers,
        args.allow_index_transfer,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
