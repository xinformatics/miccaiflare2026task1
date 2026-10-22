#!/usr/bin/env python3
"""Evaluate one GF-Screen checkpoint on the locked internal validation cohort.

This is deliberately separate from ``GF_RL_train.py``.  It never constructs an
optimizer, advances a scheduler, saves model weights, logs to W&B, or reads the
fixed-public/hidden challenge sets.  The output is an atomic JSON audit that can
be used to build a dense post-hoc learning curve from retained epoch files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from GF_RL_train import parser as training_parser
from GF_RL_trainer import val_epoch
from models.GF_Screen_RL import GFScreen_RL
from utils.data_utils import get_loader


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(
    "/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
)
# Match the immutable 2026-08-25 long-Focus invocation explicitly instead of
# relying on entrypoint defaults that may change later.  ``sw_batch_size`` and
# ``batch_size`` do not currently alter ``valid_with_labels`` (its inference
# microbatch is fixed at two), but keeping the reconstructed model arguments
# identical makes that invariant auditable rather than accidental. Validation
# Focus microbatch one preserves the same predictions while accommodating the
# largest held-out crop set on a 48-GiB GPU.
LOCKED_TRAINING_SEED = 20260825
LOCKED_BATCH_SIZE = 4
LOCKED_SW_BATCH_SIZE = 2


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score one retained GF-Screen checkpoint on internal validation"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--validation-focus-batch-size", default=1, type=int)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, type=Path)
    parser.add_argument(
        "--abdomen-json", default=ROOT / "jsons/local/abdomen.json", type=Path
    )
    parser.add_argument(
        "--chest-json", default=ROOT / "jsons/local/chest.json", type=Path
    )
    parser.add_argument(
        "--healthy-json",
        default=ROOT / "jsons/local/no_proxy_healthy.json",
        type=Path,
    )
    parser.add_argument(
        "--harmonize-cached-chest-window",
        action="store_true",
        help="must match the evaluated training recipe; disabled for the 2026-08-25 Focus run",
    )
    return parser


def validate_inputs(cli: argparse.Namespace) -> None:
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {cli.checkpoint}")
    if cli.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {cli.output}")
    if cli.workers < 0:
        raise ValueError("--workers must be non-negative")
    if cli.validation_focus_batch_size < 1:
        raise ValueError("--validation-focus-batch-size must be positive")
    for name in ("abdomen_json", "chest_json", "healthy_json"):
        path = getattr(cli, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if not cli.cache_dir.is_dir():
        raise FileNotFoundError(f"missing cache directory: {cli.cache_dir}")


def evaluation_args(cli: argparse.Namespace) -> argparse.Namespace:
    # Reuse the model/data defaults from the authoritative training entrypoint,
    # then override only paths and evaluation-runtime controls.
    args = training_parser.parse_args([])
    args.checkpoint = str(cli.checkpoint.resolve())
    args.data_dir = str(ROOT)
    args.cache_dir = str(cli.cache_dir.resolve())
    args.abdomen_json = str(cli.abdomen_json.resolve())
    args.chest_json = str(cli.chest_json.resolve())
    args.healthy_json = str(cli.healthy_json.resolve())
    args.healthy_repeat = 1
    args.samples_per_epoch = 1
    args.seed = LOCKED_TRAINING_SEED
    args.batch_size = LOCKED_BATCH_SIZE
    args.sw_batch_size = LOCKED_SW_BATCH_SIZE
    args.workers = cli.workers
    args.validation_focus_batch_size = cli.validation_focus_batch_size
    args.gpu = cli.gpu
    args.rank = cli.gpu
    args.distributed = False
    args.world_size = 1
    args.test_mode = False
    # ``GF_RL_train.main`` derives this after parsing; parsing defaults alone
    # does not create ``args.amp`` and val_epoch reads it unconditionally.
    args.amp = not bool(args.noamp)
    args.wandb_mode = "disabled"
    args.harmonize_cached_chest_window = cli.harmonize_cached_chest_window
    return args


def main() -> None:
    cli = build_cli().parse_args()
    validate_inputs(cli)
    args = evaluation_args(cli)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GF-Screen internal validation")
    if not 0 <= args.gpu < torch.cuda.device_count():
        raise ValueError(
            f"--gpu {args.gpu} is outside available range 0..{torch.cuda.device_count() - 1}"
        )

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    checkpoint = torch.load(
        cli.checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("checkpoint does not contain a state_dict")
    embedded_epoch = int(checkpoint.get("epoch", -1))
    args.max_epochs = max(1, embedded_epoch + 1)

    loader = get_loader(args)[1]
    model = GFScreen_RL(args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda(args.gpu)

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(
        include_background=False,
        reduction=MetricReduction.MEAN,
        get_not_nans=True,
    )

    started = time.time()
    dice = float(
        np.mean(
            val_epoch(
                model,
                loader,
                epoch=embedded_epoch,
                acc_func=dice_acc,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
        )
    )
    annotated_recall = dict(getattr(val_epoch, "last_diagnostics", {}))
    if not annotated_recall:
        raise RuntimeError("validation did not emit annotated-recall diagnostics")
    elapsed = time.time() - started

    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "checkpoint": {
            "path": str(cli.checkpoint.resolve()),
            "sha256": sha256_file(cli.checkpoint),
            "bytes": cli.checkpoint.stat().st_size,
            "embedded_completed_epoch": embedded_epoch,
            "best_acc": float(checkpoint.get("best_acc", 0.0)),
        },
        "internal_validation": {
            "dataset_cases": len(loader.dataset),
            "dice": dice,
            "elapsed_seconds": elapsed,
            "gpu": args.gpu,
            "workers": args.workers,
            "annotated_recall": annotated_recall,
        },
        "evaluation_source": "standalone_single_gpu_validation",
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "metric_semantics": {
            "name": "foreground_binary_sorensen_dice",
            "challenge_coefficient_equivalence": "same Dice coefficient for lesion-bearing binary masks",
            "differences_from_codabench": [
                "locked internal partial-primary labels rather than hidden labels",
                "preprocessed model-space inference before deployment postprocessing",
                "NSD is not computed by this diagnostic",
            ],
            "annotated_recall_semantics": {
                "voxel_recall": "fraction of distributed annotated foreground voxels predicted as foreground",
                "component_recall": "fraction of annotated 26-connected lesions with at least 10 percent voxel overlap",
                "partial_label_safe": True,
                "prediction_only_voxels_ignored": True,
            },
        },
        "data_contract": {
            "cache_dir": str(cli.cache_dir.resolve()),
            "abdomen_json": {
                "path": str(cli.abdomen_json.resolve()),
                "sha256": sha256_file(cli.abdomen_json),
            },
            "chest_json": {
                "path": str(cli.chest_json.resolve()),
                "sha256": sha256_file(cli.chest_json),
            },
            "healthy_json": {
                "path": str(cli.healthy_json.resolve()),
                "sha256": sha256_file(cli.healthy_json),
            },
            "harmonize_cached_chest_window": cli.harmonize_cached_chest_window,
        },
        "evaluation_recipe": {
            "seed": args.seed,
            "batch_size_per_gpu": args.batch_size,
            "sw_batch_size": args.sw_batch_size,
            "amp": bool(args.amp),
            "distributed": False,
            "validation_inference_microbatch": 2,
            "source_training_run": "gf_focus_pu_long30_e283_to313_20260825",
        },
        "role": "internal_diagnostic_only_not_checkpoint_promotion",
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json_write(cli.output, payload)
    print(
        f"INTERNAL_EVAL_COMPLETE epoch={embedded_epoch + 1} "
        f"dice={dice:.8f} cases={len(loader.dataset)} "
        f"annotated_component_recall={annotated_recall['annotated_component_recall']:.8f} "
        f"annotated_voxel_recall={annotated_recall['annotated_voxel_recall']:.8f} "
        f"elapsed_seconds={elapsed:.2f} output={cli.output}"
    )


if __name__ == "__main__":
    main()
