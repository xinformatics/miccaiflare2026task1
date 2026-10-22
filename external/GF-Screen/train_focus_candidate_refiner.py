#!/usr/bin/env python3
"""Fine-tune the pretrained GF Focus SwinUNETR on complete-label ROI patches."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from candidate_refiner_sampling import hierarchical_sample_weights, sampling_audit
from focus_candidate_refiner import (
    checkpoint_sha256,
    focus_discriminative_parameter_groups,
    focus_training_loss,
    foreground_logit,
    load_pretrained_focus,
    set_focus_encoder_trainable,
    set_focus_prompt_adapter_trainable,
)
from focus_validation_metrics import apply_case_balanced_metrics
from fault_tolerant_wandb import (
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
)
from train_candidate_refiner import configure_reproducibility, grouped_fold_rows, seed_worker


class FocusPatchDataset(Dataset):
    def __init__(
        self, rows: list[dict[str, str]], augment: bool, input_mode: str = "soft"
    ):
        if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
            raise ValueError(f"unknown Focus input mode: {input_mode}")
        self.rows = rows
        self.augment = augment
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with np.load(row["path"], allow_pickle=False) as archive:
            # The cached channels are soft-tissue CT, lung-window CT, and the
            # label-blind proposal seed, in that order.
            source = archive["image"]
            indices = {
                "soft": (0,),
                "soft_seed": (0, 2),
                "soft_lung_seed": (0, 1, 2),
            }[self.input_mode]
            image = torch.from_numpy(source[list(indices)].astype(np.float32))
            target = torch.from_numpy(archive["target"].astype(np.float32))
            label = torch.tensor(float(archive["label"]), dtype=torch.float32)
        if self.augment:
            for axis in (1, 2, 3):
                if torch.rand(()) < 0.5:
                    image = torch.flip(image, dims=(axis,))
                    target = torch.flip(target, dims=(axis,))
            if torch.rand(()) < 0.2:
                ct_channels = 2 if self.input_mode == "soft_lung_seed" else 1
                image[:ct_channels] = torch.clamp(
                    image[:ct_channels] * (0.95 + 0.10 * torch.rand(()))
                    + (-0.025 + 0.05 * torch.rand(())),
                    0.0,
                    1.0,
                )
        return image, target, label, row["patch_id"]


FOCUS_CONFIDENCE_FEATURE_NAMES = (
    "foreground_max",
    "foreground_top64_mean",
    "foreground_top512_mean",
    "foreground_fraction_0p50",
    "foreground_fraction_0p70",
    "foreground_fraction_0p90",
    "seed_probability_mean",
    "seed_probability_max",
    "prediction_0p50_seed_precision",
    "seed_coverage_0p50",
)


@torch.no_grad()
def focus_confidence_features(
    probabilities: torch.Tensor, images: torch.Tensor
) -> torch.Tensor:
    """Inference-visible candidate confidence features from Focus contours."""

    if probabilities.ndim != 5 or probabilities.shape[1] != 1:
        raise ValueError("Focus probabilities must have shape (B,1,D,H,W)")
    if images.ndim != 5 or len(images) != len(probabilities):
        raise ValueError("Focus images and probabilities must align")
    values = probabilities.float().flatten(1)
    top_count = min(512, values.shape[1])
    top = torch.topk(values, k=top_count, dim=1, sorted=True).values
    top64 = top[:, : min(64, top_count)].mean(1)
    top512 = top.mean(1)
    features = [
        top[:, 0],
        top64,
        top512,
        (values >= 0.50).float().mean(1),
        (values >= 0.70).float().mean(1),
        (values >= 0.90).float().mean(1),
    ]
    if images.shape[1] > 1:
        # Both prompt-enabled modes place the binary proposal seed last.
        seed = images[:, -1:] > 0.5
        seed_count = seed.flatten(1).sum(1).clamp_min(1)
        seed_values = probabilities.float().masked_fill(~seed, 0.0)
        seed_mean = seed_values.flatten(1).sum(1) / seed_count
        seed_max = probabilities.float().masked_fill(~seed, -1.0).flatten(1).max(1).values
        seed_max = seed_max.clamp_min(0.0)
        predicted = probabilities >= 0.50
        intersection = (predicted & seed).flatten(1).sum(1).float()
        prediction_count = predicted.flatten(1).sum(1).clamp_min(1).float()
        features.extend(
            [
                seed_mean,
                seed_max,
                intersection / prediction_count,
                intersection / seed_count.float(),
            ]
        )
    else:
        zeros = torch.zeros_like(features[0])
        features.extend([zeros, zeros, zeros, zeros])
    result = torch.stack(features, dim=1)
    if result.shape[1] != len(FOCUS_CONFIDENCE_FEATURE_NAMES):
        raise RuntimeError("Focus confidence feature count mismatch")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("non-finite Focus confidence feature")
    return result


def learning_rate(step: int, total_steps: int, warmup_steps: int, peak: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def evaluate(
    model, loader, device, segmentation_threshold_max: float = 0.80
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    threshold_steps = round((float(segmentation_threshold_max) - 0.20) / 0.05)
    if (
        not np.isfinite(segmentation_threshold_max)
        or not 0.50 <= segmentation_threshold_max <= 0.95
        or abs(0.20 + 0.05 * threshold_steps - segmentation_threshold_max) > 1e-8
    ):
        raise ValueError(
            "segmentation_threshold_max must be a 0.05 grid point in [0.50, 0.95]"
        )
    thresholds = np.linspace(
        0.20, float(segmentation_threshold_max), threshold_steps + 1
    )
    positive_ids: list[str] = []
    dice_rows: list[np.ndarray] = []
    negative_ids: list[str] = []
    negative_rows: list[np.ndarray] = []
    negative_nonempty = np.zeros(len(thresholds), dtype=np.int64)
    negative_count = 0
    all_ids: list[str] = []
    all_labels: list[float] = []
    confidence_rows: list[np.ndarray] = []
    for images, targets, labels, patch_ids in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(foreground_logit(model(images)))
        confidence_rows.append(
            focus_confidence_features(probabilities, images).cpu().numpy()
        )
        all_ids.extend(map(str, patch_ids))
        all_labels.extend(float(value) for value in labels.numpy())
        positive = labels > 0.5
        if bool(positive.any()):
            truth = targets[positive.to(device)] > 0.5
            selected_probabilities = probabilities[positive.to(device)]
            batch_rows = []
            for threshold in thresholds:
                prediction = selected_probabilities >= float(threshold)
                intersection = (prediction & truth).sum(dim=(1, 2, 3, 4)).float()
                denominator = prediction.sum(dim=(1, 2, 3, 4)) + truth.sum(
                    dim=(1, 2, 3, 4)
                )
                batch_rows.append(((2 * intersection + 1) / (denominator + 1)).cpu().numpy())
            dice_rows.append(np.stack(batch_rows, axis=1))
            indices = np.flatnonzero(positive.numpy())
            positive_ids.extend(str(patch_ids[index]) for index in indices)
        negative = ~positive
        if bool(negative.any()):
            values = probabilities[negative.to(device)]
            batch_nonempty = []
            for threshold_index, threshold in enumerate(thresholds):
                active = (values >= float(threshold)).flatten(1).any(1)
                negative_nonempty[threshold_index] += int(active.sum().cpu())
                batch_nonempty.append(active.cpu().numpy())
            negative_rows.append(np.stack(batch_nonempty, axis=1))
            indices = np.flatnonzero(negative.numpy())
            negative_ids.extend(str(patch_ids[index]) for index in indices)
            negative_count += int(negative.sum())
    grid = np.concatenate(dice_rows, axis=0).astype(np.float32)
    negative_grid = (
        np.concatenate(negative_rows, axis=0).astype(np.uint8)
        if negative_rows
        else np.zeros((0, len(thresholds)), dtype=np.uint8)
    )
    if len(negative_ids) != negative_count or negative_grid.shape != (
        negative_count,
        len(thresholds),
    ):
        raise ValueError("negative validation identity/activation mismatch")
    if not np.array_equal(negative_grid.sum(axis=0), negative_nonempty):
        raise ValueError("negative validation activation sum mismatch")
    confidence = np.concatenate(confidence_rows, axis=0).astype(np.float32)
    if (
        len(all_ids) != len(set(all_ids))
        or confidence.shape
        != (len(all_ids), len(FOCUS_CONFIDENCE_FEATURE_NAMES))
        or len(all_labels) != len(all_ids)
    ):
        raise ValueError("Focus confidence feature identity/shape mismatch")
    means = grid.mean(axis=0)
    best = max(
        range(len(thresholds)),
        key=lambda index: (float(means[index]), -abs(float(thresholds[index]) - 0.5)),
    )
    metrics = {
        "segmentation_threshold": float(thresholds[best]),
        "segmentation_dice": float(means[best]),
        "negative_patches": int(negative_count),
        "negative_nonempty_rate": float(negative_nonempty[best] / max(negative_count, 1)),
        "selection_score": float(means[best] - 0.02 * negative_nonempty[best] / max(negative_count, 1)),
    }
    return metrics, {
        "positive_patch_id": np.asarray(positive_ids),
        "segmentation_thresholds": thresholds.astype(np.float32),
        "segmentation_dice_grid": grid,
        "negative_patch_id": np.asarray(negative_ids),
        "negative_nonempty_grid": negative_grid,
        "negative_nonempty_by_threshold": negative_nonempty,
        "all_patch_id": np.asarray(all_ids),
        "all_label": np.asarray(all_labels, dtype=np.float32),
        "confidence_feature_names": np.asarray(FOCUS_CONFIDENCE_FEATURE_NAMES),
        "confidence_features": confidence,
    }


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--loss-recipe",
        choices=("boundary_balanced", "release_dice_ce"),
        default="boundary_balanced",
        help="controlled contour objective; release_dice_ce matches GF training",
    )
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
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
        help="bounded weighted samples per epoch; zero uses every training-row count",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=1.0,
        help=(
            "learning-rate multiplier for trainable pretrained encoder tensors; "
            "1.0 preserves the historical single-rate optimizer"
        ),
    )
    parser.add_argument(
        "--segmentation-threshold-max",
        type=float,
        default=0.80,
        help="upper endpoint of the 0.05 validation contour grid",
    )
    parser.add_argument(
        "--input-mode",
        choices=("soft", "soft_seed", "soft_lung_seed"),
        default="soft",
        help=(
            "released soft-tissue CT input, CT plus seed, or soft/lung CT "
            "windows plus a label-blind proposal seed"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--wandb-project", default="pancancer-ct")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.workers < 0
        or args.lr <= 0
        or not 0.0 < args.encoder_lr_scale <= 1.0
        or args.samples_per_epoch < 0
        or not 0.50 <= args.segmentation_threshold_max <= 0.95
        or abs(
            0.20
            + 0.05 * round((args.segmentation_threshold_max - 0.20) / 0.05)
            - args.segmentation_threshold_max
        )
        > 1e-8
    ):
        parser.error("invalid training configuration")
    configure_reproducibility(args.seed + args.fold)
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train_rows, validation_rows = grouped_fold_rows(rows, args.folds, args.fold, args.seed)
    weights = hierarchical_sample_weights(
        train_rows, args.sampling_temperature, args.balance_source_groups
    )
    sampler_generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader_generator = torch.Generator().manual_seed(
        args.seed + args.fold + 10_000
    )
    validation_loader_generator = torch.Generator().manual_seed(
        args.seed + args.fold + 20_000
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        args.samples_per_epoch or len(train_rows),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        FocusPatchDataset(train_rows, True, args.input_mode), batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )
    validation_loader = DataLoader(
        FocusPatchDataset(validation_rows, False, args.input_mode), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=validation_loader_generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_channels = {
        "soft": 1,
        "soft_seed": 2,
        "soft_lung_seed": 3,
    }[args.input_mode]
    model = load_pretrained_focus(
        args.pretrained, use_checkpoint=True, in_channels=input_channels
    )
    if args.freeze_encoder:
        set_focus_encoder_trainable(model, False)
        if args.input_mode in {"soft_seed", "soft_lung_seed"}:
            # The zero-initialized prompt channel must remain learnable even
            # when the released CT encoder is frozen.
            set_focus_prompt_adapter_trainable(model, True)
    model.to(device)
    parameter_groups = focus_discriminative_parameter_groups(
        model, args.encoder_lr_scale
    )
    trainable = [
        parameter for group in parameter_groups for parameter in group["params"]
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(len(train_loader), total_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    index_digest = hashlib.sha256(args.index.read_bytes()).hexdigest()
    pretrained_digest = checkpoint_sha256(args.pretrained)
    audit = sampling_audit(
        train_rows, args.sampling_temperature, args.balance_source_groups
    )
    audit.update({
        "index": str(args.index.resolve()), "index_sha256": index_digest,
        "pretrained": str(args.pretrained.resolve()), "pretrained_sha256": pretrained_digest,
        "fold": args.fold, "folds": args.folds,
        "samples_per_epoch": args.samples_per_epoch or len(train_rows),
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "extra_channel_initialization": (
            "zero; released CT channel preserved"
            if input_channels > 1
            else "released checkpoint unchanged"
        ),
    })
    (args.output / "sampling_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    run = None
    if args.wandb_name:
        wandb_run_id = hashlib.sha256(
            f"{args.output.resolve()}::{args.fold}".encode("utf-8")
        ).hexdigest()[:16]
        run = safe_wandb_init(
            project=args.wandb_project, name=args.wandb_name,
            id=wandb_run_id, resume="allow",
            dir=str(args.output), config=vars(args),
        )
    best = -float("inf")
    history = []
    global_step = 0
    start_epoch = 1
    last_path = args.output / "checkpoint_last.pt"
    history_path = args.output / "history.json"
    if args.resume and last_path.is_file():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("index_sha256") != index_digest:
            raise ValueError("cannot resume with a different patch index")
        if checkpoint.get("pretrained_sha256") != pretrained_digest:
            raise ValueError("cannot resume with a different pretrained checkpoint")
        if int(checkpoint.get("fold", -1)) != args.fold:
            raise ValueError("cannot resume a different grouped fold")
        saved_config = checkpoint.get("config", {})
        legacy_defaults = {
            "segmentation_threshold_max": 0.80,
            "encoder_lr_scale": 1.0,
        }
        for key in (
            "folds", "epochs", "batch_size", "workers", "lr", "weight_decay",
            "loss_recipe", "sampling_temperature", "samples_per_epoch",
            "freeze_encoder", "encoder_lr_scale", "input_mode",
            "segmentation_threshold_max", "seed",
        ):
            if saved_config.get(key, legacy_defaults.get(key)) != getattr(args, key):
                raise ValueError(f"cannot resume with changed training option: {key}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(
            checkpoint.get("global_step", (start_epoch - 1) * len(train_loader))
        )
        best = float(checkpoint.get("best_score", -float("inf")))
        generator_states = checkpoint.get("generator_states", {})
        if generator_states:
            sampler_generator.set_state(generator_states["sampler"])
            train_loader_generator.set_state(generator_states["train_loader"])
            validation_loader_generator.set_state(
                generator_states["validation_loader"]
            )
        if isinstance(checkpoint.get("history"), list):
            history = checkpoint["history"]
        elif history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"resuming Focus fold {args.fold} at epoch {start_epoch}", flush=True)
    if start_epoch == 1 and not history:
        baseline_started = time.time()
        metrics, predictions = evaluate(
            model,
            validation_loader,
            device,
            segmentation_threshold_max=args.segmentation_threshold_max,
        )
        metrics = apply_case_balanced_metrics(metrics, predictions, validation_rows)
        record = {
            "epoch": 0,
            "train_loss": None,
            "lr": 0.0,
            "seconds": time.time() - baseline_started,
            "state": "pretrained_before_finetuning",
            **metrics,
        }
        history.append(record)
        best = float(metrics["selection_score"])
        checkpoint = {
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": 0, "fold": args.fold,
            "folds": args.folds, "config": vars(args), "metrics": metrics,
            "index_sha256": index_digest, "pretrained_sha256": pretrained_digest,
            "architecture": f"GF Focus SwinUNETR contour refiner ({args.input_mode})",
            "loss_recipe": args.loss_recipe,
            "best_score": best, "global_step": global_step, "history": history,
            "generator_states": {
                "sampler": sampler_generator.get_state(),
                "train_loader": train_loader_generator.get_state(),
                "validation_loader": validation_loader_generator.get_state(),
            },
        }
        torch.save(checkpoint, last_path)
        torch.save(checkpoint, args.output / "checkpoint_best.pt")
        atomic_npz(args.output / "validation_predictions_best.npz", predictions)
        # Preserve every scored state so proposal families can select their
        # checkpoint epoch using only grouped held-out predictions.  This is
        # an artifact-only operation and does not alter optimization or RNG.
        torch.save(checkpoint, args.output / "checkpoint_epoch_0000.pt")
        atomic_npz(
            args.output / "validation_predictions_epoch_0000.npz", predictions
        )
        history_path.write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
        if run is not None:
            run = safe_wandb_log(run, record, step=0)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); running = 0.0; started = time.time()
        for images, targets, labels, _ in train_loader:
            lr = learning_rate(global_step, total_steps, warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr * float(group.get("lr_scale", 1.0))
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _ = focus_training_loss(
                    model(images), targets, labels, args.loss_recipe
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            scaler.step(optimizer); scaler.update()
            running += float(loss.detach()); global_step += 1
        metrics, predictions = evaluate(
            model,
            validation_loader,
            device,
            segmentation_threshold_max=args.segmentation_threshold_max,
        )
        metrics = apply_case_balanced_metrics(metrics, predictions, validation_rows)
        record = {
            "epoch": epoch, "train_loss": running / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - started,
            **metrics,
        }
        if len(optimizer.param_groups) > 1:
            record["encoder_lr"] = optimizer.param_groups[1]["lr"]
        history.append(record); print(json.dumps(record, sort_keys=True), flush=True)
        if run is not None:
            run = safe_wandb_log(run, record, step=epoch)
        checkpoint = {
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "fold": args.fold,
            "folds": args.folds, "config": vars(args), "metrics": metrics,
            "index_sha256": index_digest, "pretrained_sha256": pretrained_digest,
            "architecture": f"GF Focus SwinUNETR contour refiner ({args.input_mode})",
            "loss_recipe": args.loss_recipe,
            "best_score": max(best, float(metrics["selection_score"])),
            "global_step": global_step,
            "history": history,
            "generator_states": {
                "sampler": sampler_generator.get_state(),
                "train_loader": train_loader_generator.get_state(),
                "validation_loader": validation_loader_generator.get_state(),
            },
        }
        torch.save(checkpoint, last_path)
        torch.save(
            checkpoint, args.output / f"checkpoint_epoch_{epoch:04d}.pt"
        )
        atomic_npz(
            args.output / f"validation_predictions_epoch_{epoch:04d}.npz",
            predictions,
        )
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            torch.save(checkpoint, args.output / "checkpoint_best.pt")
            atomic_npz(args.output / "validation_predictions_best.npz", predictions)
        history_path.write_text(json.dumps(history, indent=2) + "\n")
    (args.output / "COMPLETE.txt").write_text(
        __import__("datetime").datetime.now().astimezone().isoformat() + "\n"
    )
    safe_wandb_finish(run)


if __name__ == "__main__":
    main()
