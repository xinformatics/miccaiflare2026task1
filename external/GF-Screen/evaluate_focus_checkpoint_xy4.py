#!/usr/bin/env python3
"""Evaluate the complete four-view x/y flip group on one held-out Focus fold."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_focus_checkpoint_tta import INPUT_CHANNELS, sha256
from focus_candidate_refiner import foreground_logit, load_pretrained_focus
from focus_validation_metrics import apply_case_balanced_metrics
from train_candidate_refiner import configure_reproducibility, grouped_fold_rows, seed_worker
from train_focus_candidate_refiner import FocusPatchDataset, atomic_npz, evaluate


FLIP_SETS = ((), (2,), (3,), (2, 3))


class FocusXY4ProbabilityAsLogits(torch.nn.Module):
    """Average the identity, x, y, and joint x+y restored probabilities."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def _probability(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(foreground_logit(self.model(images))).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probability = torch.zeros(
            (images.shape[0], 1, *images.shape[2:]),
            dtype=torch.float32,
            device=images.device,
        )
        for dimensions in FLIP_SETS:
            view = torch.flip(images, dims=dimensions) if dimensions else images
            restored = self._probability(view)
            if dimensions:
                restored = torch.flip(restored, dims=dimensions)
            probability = probability + restored
        probability = probability / len(FLIP_SETS)
        foreground = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


def run(args: argparse.Namespace) -> dict[str, object]:
    configure_reproducibility(args.seed + args.fold)
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty Focus patch index")
    _, validation_rows = grouped_fold_rows(rows, args.folds, args.fold, args.seed)
    index_digest = sha256(args.index)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False, mmap=True
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
        generator=torch.Generator().manual_seed(args.seed + args.fold + 40_000),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_pretrained_focus(
        args.checkpoint,
        use_checkpoint=True,
        in_channels=INPUT_CHANNELS[input_mode],
    ).eval().to(device)
    metrics, predictions = evaluate(
        FocusXY4ProbabilityAsLogits(base).eval(),
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
        "tta_recipe": "complete_xy_flip_group",
        "tta_flip_sets": [list(dimensions) for dimensions in FLIP_SETS],
        "tta_views": len(FLIP_SETS),
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--segmentation-threshold-max", type=float, default=0.95)
    args = parser.parse_args()
    if args.folds < 2 or not 0 <= args.fold < args.folds or args.batch_size < 1 or args.workers < 0:
        parser.error("invalid fold or loader configuration")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
