#!/usr/bin/env python3
"""CPU-only construction/load proof for the deployment STU-Net checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import torch

from predict_stunet_anchor_sparse import AnchorSparsePredictor
from predict_stunet_safe import safe_initialize, sha256_file


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--terminal", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.model_dir / "fold_all" / args.checkpoint
    before = sha256_file(checkpoint)
    if before != args.expected_checkpoint_sha256:
        raise RuntimeError("STU-Net deployment checkpoint identity mismatch")
    device_keyword = (
        "perform_everything_on_device"
        if "perform_everything_on_device" in inspect.signature(AnchorSparsePredictor).parameters
        else "perform_everything_on_gpu"
    )
    predictor = AnchorSparsePredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
        **{device_keyword: False},
    )
    started = time.perf_counter()
    safe_initialize(predictor, args.model_dir, args.checkpoint)
    load_seconds = time.perf_counter() - started
    after = sha256_file(checkpoint)
    if after != before:
        raise RuntimeError("STU-Net deployment checkpoint changed while loading")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "CHECKPOINT_LOAD_COMPATIBLE",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "nnunetv2_version": importlib.metadata.version("nnunetv2"),
        "trainer_name": predictor.trainer_name,
        "configuration": predictor.configuration_manager.configuration[
            "data_identifier"
        ],
        "parameters": sum(value.numel() for value in predictor.network.parameters()),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": after,
            "bytes": checkpoint.stat().st_size,
        },
        "model_initialization_seconds": load_seconds,
        "public_images_read": False,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "deployment_authorized": False,
    }
    atomic_json(args.terminal, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
