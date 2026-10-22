#!/usr/bin/env python3
"""Isolated CUDA smoke test for the corrected ResEnc S6 training stack.

This intentionally does not call ``on_train_start`` and cannot write a model
checkpoint.  It builds the production trainer, runs one autocast
forward/backward/optimizer step on a synthetic partial-box target, and records
the observed peak memory.  ``CUDA_VISIBLE_DEVICES`` should expose exactly one
otherwise-idle physical GPU when this script is invoked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.batch_size != 8:
        raise SystemExit("production rank-local batch-size must be exactly 8")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise SystemExit("expose exactly one physical GPU with CUDA_VISIBLE_DEVICES")
    os.environ["nnUNet_raw"] = str(root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(output.parent / "isolated_results")
    os.environ["nnUNet_compile"] = "false"
    os.environ["nnUNet_n_proc_DA"] = "0"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch

    from nnunet.resenc_s6_partialbox import (
        nnUNetTrainerResEncS6PanCancerPartialBox,
    )

    plans_path = root / "nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json"
    dataset_path = root / "nnUNet_preprocessed/Dataset501_PanCancer/dataset.json"
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    print("constructing production trainer", flush=True)
    trainer = nnUNetTrainerResEncS6PanCancerPartialBox(
        plans,
        "3d_fullres_ResEncS6",
        "all",
        dataset,
        device=torch.device("cuda", 0),
    )
    trainer.initialize()
    network = trainer.network
    network.train()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    if parameter_count != trainer.expected_parameter_count:
        raise RuntimeError(
            f"parameter count changed: {parameter_count} != "
            f"{trainer.expected_parameter_count}"
        )
    patch = tuple(int(value) for value in trainer.configuration_manager.patch_size)
    if patch != (128, 160, 160):
        raise RuntimeError(f"patch size changed: {patch}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    data = torch.zeros(
        (args.batch_size, 1, *patch),
        dtype=torch.float32,
        device="cuda:0",
    )
    trainer.optimizer.zero_grad(set_to_none=True)
    print(
        f"running batch={args.batch_size} patch={patch} parameters={parameter_count}",
        flush=True,
    )
    with torch.autocast("cuda", enabled=True):
        prediction = network(data)
        predictions = prediction if isinstance(prediction, (list, tuple)) else [prediction]
        targets = []
        for value in predictions:
            spatial = tuple(int(length) for length in value.shape[2:])
            target = torch.full(
                (args.batch_size, 1, *spatial),
                2,
                dtype=torch.int16,
                device="cuda:0",
            )
            center = tuple(length // 2 for length in spatial)
            trusted_half = tuple(max(2, length // 8) for length in spatial)
            lesion_half = tuple(max(1, length // 32) for length in spatial)
            trusted = tuple(
                slice(mid - half, mid + half)
                for mid, half in zip(center, trusted_half, strict=True)
            )
            lesion = tuple(
                slice(mid - half, mid + half)
                for mid, half in zip(center, lesion_half, strict=True)
            )
            target[(slice(None), 0, *trusted)] = 0
            target[(slice(None), 0, *lesion)] = 1
            targets.append(target)
        loss = trainer.loss(prediction, targets if len(targets) > 1 else targets[0])

    print(f"forward complete loss={float(loss.detach()):.6f}", flush=True)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke-test loss")
    if trainer.grad_scaler is not None:
        trainer.grad_scaler.scale(loss).backward()
        trainer.grad_scaler.unscale_(trainer.optimizer)
        trainer.grad_scaler.step(trainer.optimizer)
        trainer.grad_scaler.update()
    else:
        loss.backward()
        trainer.optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "physical_gpu": visible,
        "architecture": trainer.architecture_name,
        "parameter_count": parameter_count,
        "rank_local_batch_size": args.batch_size,
        "patch_size": list(patch),
        "loss": float(loss.detach().cpu()),
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "checkpoint_created": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
