#!/usr/bin/env python3
"""Evaluate flip-TTA for one case-held-out Focus checkpoint.

This is deliberately patch-OOF only: it reads the training patch index and a
checkpoint belonging to the requested held-out fold, writes prediction grids,
and never reads public/hidden cases or creates a submission.
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

from focus_candidate_refiner import foreground_logit, load_pretrained_focus
from focus_validation_metrics import apply_case_balanced_metrics
from train_candidate_refiner import configure_reproducibility, grouped_fold_rows, seed_worker
from train_focus_candidate_refiner import FocusPatchDataset, atomic_npz, evaluate


AXIS_TO_DIMENSION = {"x": 2, "y": 3, "z": 4}
INPUT_CHANNELS = {"soft": 1, "soft_seed": 2, "soft_lung_seed": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_tta_axes(value: str) -> tuple[int, ...]:
    normalized = value.lower().replace(",", "").replace(" ", "")
    if any(axis not in AXIS_TO_DIMENSION for axis in normalized):
        raise ValueError("TTA axes must contain only x, y, and z")
    if len(set(normalized)) != len(normalized):
        raise ValueError("TTA axes cannot repeat an axis")
    return tuple(AXIS_TO_DIMENSION[axis] for axis in normalized)


class FocusTTAProbabilityAsLogits(torch.nn.Module):
    """Expose averaged foreground probabilities through the legacy logit API."""

    def __init__(self, model: torch.nn.Module, flip_dimensions: tuple[int, ...]):
        super().__init__()
        if len(set(flip_dimensions)) != len(flip_dimensions) or any(
            dimension not in AXIS_TO_DIMENSION.values()
            for dimension in flip_dimensions
        ):
            raise ValueError("invalid or repeated Focus TTA dimensions")
        self.model = model
        self.flip_dimensions = flip_dimensions

    def _probability(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(foreground_logit(self.model(images))).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probability = self._probability(images)
        for dimension in self.flip_dimensions:
            augmented = torch.flip(images, dims=(dimension,))
            restored = torch.flip(
                self._probability(augmented), dims=(dimension,)
            )
            probability = probability + restored
        probability = probability / (1 + len(self.flip_dimensions))
        # ``evaluate`` consumes two-class logits through foreground_logit and
        # sigmoid. A zero background logit plus logit(p) foreground therefore
        # preserves the already averaged probability exactly (up to the
        # bounded numerical conversion).
        foreground = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


def run(args: argparse.Namespace) -> dict[str, object]:
    flip_dimensions = parse_tta_axes(args.tta_axes)
    configure_reproducibility(args.seed + args.fold)
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty Focus patch index")
    _, validation_rows = grouped_fold_rows(rows, args.folds, args.fold, args.seed)
    index_digest = sha256(args.index)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = checkpoint.get("config", {})
    input_mode = str(config.get("input_mode", ""))
    if (
        checkpoint.get("index_sha256") != index_digest
        or int(checkpoint.get("fold", -1)) != args.fold
        or int(checkpoint.get("folds", -1)) != args.folds
        or int(config.get("seed", -1)) != args.seed
        or input_mode not in INPUT_CHANNELS
    ):
        raise ValueError("Focus checkpoint/index/fold identity mismatch")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 0:
        raise ValueError("invalid Focus checkpoint epoch")
    del checkpoint

    loader = DataLoader(
        FocusPatchDataset(validation_rows, False, input_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed + args.fold + 30_000),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_pretrained_focus(
        args.checkpoint,
        use_checkpoint=True,
        in_channels=INPUT_CHANNELS[input_mode],
    ).eval().to(device)
    model = FocusTTAProbabilityAsLogits(base, flip_dimensions).eval()
    metrics, predictions = evaluate(
        model,
        loader,
        device,
        segmentation_threshold_max=args.segmentation_threshold_max,
    )
    metrics = apply_case_balanced_metrics(metrics, predictions, validation_rows)
    args.prediction.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.prediction, predictions)
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": "one case-grouped internal held-out fold; diagnostic only",
        "fold": args.fold,
        "folds": args.folds,
        "epoch": epoch,
        "seed": args.seed,
        "held_out_cases": len({str(row["case"]) for row in validation_rows}),
        "tta_axes": args.tta_axes.lower().replace(",", "").replace(" ", ""),
        "tta_views": 1 + len(flip_dimensions),
        "input_mode": input_mode,
        "index": str(args.index.resolve()),
        "index_sha256": index_digest,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction": str(args.prediction.resolve()),
        "prediction_sha256": sha256(args.prediction),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": sha256(Path(__file__)),
        "metrics": metrics,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_name(f".{args.summary.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--folds", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tta-axes", default="xy")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--segmentation-threshold-max", type=float, default=0.95)
    args = parser.parse_args()
    if (
        args.folds < 2
        or not 0 <= args.fold < args.folds
        or args.batch_size < 1
        or args.workers < 0
    ):
        parser.error("invalid fold or loader configuration")
    try:
        parse_tta_axes(args.tta_axes)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
