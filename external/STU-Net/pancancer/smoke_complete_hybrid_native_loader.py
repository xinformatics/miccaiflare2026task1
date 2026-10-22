#!/usr/bin/env python3
"""Smoke completeness-aware sampling and targets through nnU-Net's loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancerCompleteHybrid import (  # noqa: E402
    STUNetTrainerPanCancerComponentBoxE160CompleteHybrid,
    complete_hybrid_target,
    complete_priority_loader_factory,
    complete_priority_probabilities,
    is_complete_identifier,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader  # noqa: E402
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-folder", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE" or audit.get("decision") != "IMPLEMENTED_NOT_LAUNCHED":
        raise ValueError("unexpected complete-hybrid audit state")
    identifiers = sorted(path.stem for path in args.preprocessed_folder.glob("*.pkl"))
    trainer = STUNetTrainerPanCancerComponentBoxE160CompleteHybrid
    if len(identifiers) != trainer.expected_case_count:
        raise ValueError("native-loader case count differs from trainer contract")
    inventory_sha = hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest()
    if inventory_sha != trainer.expected_identifier_sha256:
        raise ValueError("native-loader identifier SHA differs from trainer contract")
    dataset = nnUNetDatasetBlosc2(str(args.preprocessed_folder), identifiers)
    dataset_json = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    plans_manager = PlansManager(str(args.plans))
    label_manager = plans_manager.get_label_manager(dataset_json)
    factory = complete_priority_loader_factory(nnUNetDataLoader)
    common = {
        "batch_size": 2,
        "patch_size": [128, 160, 160],
        "final_patch_size": [128, 160, 160],
        "label_manager": label_manager,
        "oversample_foreground_percent": 0.5,
        "sampling_probabilities": None,
        "pad_sides": None,
        "probabilistic_oversampling": False,
        "transforms": None,
    }
    training_loader = factory(dataset, **common)
    observed = np.asarray(training_loader.sampling_probabilities, dtype=np.float64)
    expected = complete_priority_probabilities(identifiers)
    maximum_probability_error = float(np.max(np.abs(observed - expected)))
    if maximum_probability_error != 0.0:
        raise ValueError(f"native sampling mismatch: {maximum_probability_error}")
    validation_loader = factory(dataset, **common)
    if validation_loader.sampling_probabilities is not None:
        raise ValueError("validation loader was unexpectedly reweighted")

    complete_key = next(key for key in identifiers if is_complete_identifier(key))
    partial_key = next(key for key in identifiers if not is_complete_identifier(key))
    training_loader.get_indices = lambda: [complete_key, partial_key]
    batch = training_loader.generate_train_batch()
    if list(batch["keys"]) != [complete_key, partial_key]:
        raise ValueError("native batch keys were not preserved")
    target = batch["target"]
    if isinstance(target, list):
        raise ValueError("transform-free native smoke unexpectedly returned deep supervision")
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    spacing = tuple(
        float(value)
        for value in plans_manager.get_configuration("3d_fullres").spacing
    )
    margins = tuple(max(1, int(np.ceil(16.0 / value))) for value in spacing)
    converted = complete_hybrid_target(
        target, batch["keys"], tuple(target.shape[2:]), margins
    )
    source_complete = target[0]
    known_complete = (source_complete == 0) | (source_complete == 1)
    if not np.array_equal(
        converted[0][known_complete].cpu().numpy(),
        source_complete[known_complete].cpu().numpy(),
    ):
        raise ValueError("complete native target was not preserved")
    if int((converted[1] == 2).sum()) == 0 or int((converted[1] == 1).sum()) == 0:
        raise ValueError("partial native target lacks foreground or ignored background")

    complete = np.asarray([is_complete_identifier(value) for value in identifiers])
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "native_nnunet_complete_hybrid_loader_and_target_smoke_no_training",
        "gpu_used": False,
        "public_or_hidden_labels_read": False,
        "training_case_count": len(identifiers),
        "complete_case_count": int(complete.sum()),
        "partial_case_count": int((~complete).sum()),
        "identifier_inventory_sha256": inventory_sha,
        "native_loader_class": type(training_loader).__name__,
        "data_shape": list(training_loader.data_shape),
        "segmentation_shape": list(training_loader.seg_shape),
        "training_probability_sum": float(observed.sum()),
        "training_probability_minimum": float(observed.min()),
        "training_probability_maximum": float(observed.max()),
        "maximum_probability_absolute_error": maximum_probability_error,
        "validation_sampling_probabilities": None,
        "forced_batch_keys": [complete_key, partial_key],
        "complete_valid_voxels": int((converted[0] != 2).sum()),
        "partial_valid_voxels": int((converted[1] != 2).sum()),
        "partial_foreground_voxels": int((converted[1] == 1).sum()),
        "inputs": {
            "trainer_source": {"path": str(args.trainer_source.resolve()), "sha256": sha256_file(args.trainer_source)},
            "runner_source": {"path": str(args.runner_source.resolve()), "sha256": sha256_file(args.runner_source)},
            "audit": {"path": str(args.audit.resolve()), "sha256": sha256_file(args.audit)},
            "plans": {"path": str(args.plans.resolve()), "sha256": sha256_file(args.plans)},
            "dataset_json": {"path": str(args.dataset_json.resolve()), "sha256": sha256_file(args.dataset_json)}
        },
        "submission_created": False,
        "uploaded": False
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    print(f"terminal_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
