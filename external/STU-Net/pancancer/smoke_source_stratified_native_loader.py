#!/usr/bin/env python3
"""Smoke the size-preserving sampler through nnU-Net's native data loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancerSourceBalanced import (  # noqa: E402
    STUNetTrainerPanCancerComponentBoxE160SourceStratified,
    source_balanced_loader_factory,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader  # noqa: E402
from nnunetv2.training.dataloading.nnunet_dataset import (  # noqa: E402
    nnUNetDatasetBlosc2,
)
from nnunetv2.utilities.plans_handling.plans_handler import (  # noqa: E402
    PlansManager,
)


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
    if audit.get("status") != "analysis_only_not_launched":
        raise ValueError("unexpected source-stratified audit status")
    identifiers = sorted(
        path.stem for path in args.preprocessed_folder.glob("*.pkl")
    )
    expected_identifiers = [row["identifier"] for row in audit["identifiers"]]
    if identifiers != expected_identifiers:
        raise ValueError("native-loader identifier order differs from audit")
    dataset = nnUNetDatasetBlosc2(str(args.preprocessed_folder), identifiers)
    dataset_json = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    label_manager = PlansManager(str(args.plans)).get_label_manager(dataset_json)
    factory = source_balanced_loader_factory(nnUNetDataLoader)
    common = {
        "batch_size": 2,
        "patch_size": [128, 160, 160],
        "final_patch_size": [128, 160, 160],
        "label_manager": label_manager,
        "oversample_foreground_percent": 1.0,
        "sampling_probabilities": None,
        "pad_sides": None,
        "probabilistic_oversampling": False,
        "transforms": None,
    }
    training_loader = factory(dataset, **common)
    observed = np.asarray(training_loader.sampling_probabilities, dtype=np.float64)
    expected = np.asarray(
        [row["sampling_probability"] for row in audit["identifiers"]],
        dtype=np.float64,
    )
    maximum_absolute_error = float(np.max(np.abs(observed - expected)))
    if maximum_absolute_error != 0.0:
        raise ValueError(
            f"native-loader probability audit mismatch: {maximum_absolute_error}"
        )
    validation_loader = factory(dataset, **common)
    if validation_loader.sampling_probabilities is not None:
        raise ValueError("validation loader was unexpectedly reweighted")
    expected_shape = (2, 1, 128, 160, 160)
    if tuple(training_loader.data_shape) != expected_shape:
        raise ValueError(f"unexpected native data shape: {training_loader.data_shape}")
    if tuple(training_loader.seg_shape) != expected_shape:
        raise ValueError(f"unexpected native segmentation shape: {training_loader.seg_shape}")
    trainer = STUNetTrainerPanCancerComponentBoxE160SourceStratified
    if len(identifiers) != trainer.expected_case_count:
        raise ValueError("trainer expected-case contract mismatch")
    payload = {
        "schema_version": 1,
        "status": "COMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "native_nnunet_loader_contract_smoke_no_training",
        "gpu_used": False,
        "public_or_hidden_labels_read": False,
        "training_case_count": len(identifiers),
        "trainer_name": trainer.__name__,
        "source_fraction": trainer.source_fraction,
        "raw_per_case_ratio_cap": trainer.raw_ratio_cap,
        "identifier_inventory_sha256": trainer.expected_identifier_sha256,
        "size_bin_counts": list(trainer.expected_size_bin_counts),
        "native_loader_class": type(training_loader).__name__,
        "data_shape": list(training_loader.data_shape),
        "segmentation_shape": list(training_loader.seg_shape),
        "training_probability_sum": float(observed.sum()),
        "training_probability_minimum": float(observed.min()),
        "training_probability_maximum": float(observed.max()),
        "maximum_probability_absolute_error_vs_audit": maximum_absolute_error,
        "validation_sampling_probabilities": None,
        "inputs": {
            "trainer_source": {
                "path": str(args.trainer_source.resolve()),
                "sha256": sha256_file(args.trainer_source),
            },
            "runner_source": {
                "path": str(args.runner_source.resolve()),
                "sha256": sha256_file(args.runner_source),
            },
            "audit": {
                "path": str(args.audit.resolve()),
                "sha256": sha256_file(args.audit),
            },
            "plans": {
                "path": str(args.plans.resolve()),
                "sha256": sha256_file(args.plans),
            },
            "dataset_json": {
                "path": str(args.dataset_json.resolve()),
                "sha256": sha256_file(args.dataset_json),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    print(f"terminal_sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
