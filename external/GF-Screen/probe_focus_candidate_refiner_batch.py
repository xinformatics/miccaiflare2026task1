#!/usr/bin/env python3
"""Probe a real Focus optimizer step before committing a five-fold batch size."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from focus_candidate_refiner import (
    checkpoint_sha256,
    focus_training_loss,
    load_pretrained_focus,
    set_focus_encoder_trainable,
)
from train_candidate_refiner import grouped_fold_rows
from train_focus_candidate_refiner import FocusPatchDataset


def run(args: argparse.Namespace) -> dict:
    rows = __import__("csv").DictReader(args.index.open(newline="", encoding="utf-8"))
    all_rows = list(rows)
    train_rows, _ = grouped_fold_rows(all_rows, args.folds, args.fold, args.seed)
    positive_rows = [row for row in train_rows if int(row["label"]) == 1]
    if len(positive_rows) < args.batch_size:
        raise ValueError("not enough positive patches for worst-case batch probe")
    dataset = FocusPatchDataset(positive_rows[: args.batch_size], augment=False)
    batch = [dataset[index] for index in range(args.batch_size)]
    images = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a batch-memory probe")
    device = torch.device("cuda")
    model = load_pretrained_focus(args.pretrained, use_checkpoint=True)
    set_focus_encoder_trainable(model, False)
    model.to(device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    images = images.to(device)
    targets = targets.to(device)
    labels = labels.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=True):
        loss, _ = focus_training_loss(
            model(images), targets, labels, "boundary_balanced"
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(trainable, 5.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)
    properties = torch.cuda.get_device_properties(device)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "FIT",
        "batch_size": args.batch_size,
        "probe_loss_recipe": "boundary_balanced",
        "probe_patches": args.batch_size,
        "all_positive_worst_case": True,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "device": properties.name,
        "device_total_mib": properties.total_memory / 2**20,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "index": str(args.index.resolve()),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "pretrained": str(args.pretrained.resolve()),
        "pretrained_sha256": checkpoint_sha256(args.pretrained),
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "public_read": False,
        "hidden_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    try:
        payload = run(args)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error).lower():
            raise
        print(f"CUDA_OOM batch_size={args.batch_size}: {error}", file=sys.stderr)
        raise SystemExit(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
