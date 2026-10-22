#!/usr/bin/env python3
"""Exercise the real component-center loader without training or a GPU."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from STUNetTrainerPanCancerSourceBalanced import (
    component_center_source_balanced_loader_factory,
    foreground_proxy_counts_from_dataset,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_json = load_json(str(args.dataset_json))
    plans = PlansManager(str(args.plans))
    configuration = plans.get_configuration("3d_fullres_STUNetB_PartialBox")
    label_manager = plans.get_label_manager(dataset_json)
    dataset = nnUNetDatasetBlosc2(str(args.preprocessed))
    if len(dataset.identifiers) != 17_575:
        raise SystemExit(f"expected 17575 cases, found {len(dataset.identifiers)}")

    factory = component_center_source_balanced_loader_factory(
        nnUNetDataLoader,
        source_fraction=0.25,
        raw_ratio_cap=3.0,
        component_center_fraction=0.25,
        foreground_count_resolver=foreground_proxy_counts_from_dataset,
    )
    kwargs = {
        "batch_size": 2,
        "patch_size": configuration.patch_size,
        "final_patch_size": configuration.patch_size,
        "label_manager": label_manager,
        "oversample_foreground_percent": 1.0,
        "sampling_probabilities": None,
        "transforms": None,
    }
    training_loader = factory(dataset, **kwargs)
    validation_loader = factory(dataset, **kwargs)
    batch = training_loader.generate_train_batch()
    probabilities = np.asarray(training_loader.sampling_probabilities)
    if (
        training_loader.__class__.__name__ != "ComponentCenterLoader"
        or training_loader.lesion_uniform_center_fraction != 0.25
        or validation_loader.__class__ is not nnUNetDataLoader
        or validation_loader.sampling_probabilities is not None
        or batch["data"].shape != tuple(training_loader.data_shape)
        or batch["target"].shape != tuple(training_loader.seg_shape)
        or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise SystemExit("real component-center loader smoke failed")

    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "purpose": "native_component_center_source_loader_one_batch_no_training",
        "training_case_count": len(dataset.identifiers),
        "training_loader_class": training_loader.__class__.__name__,
        "validation_loader_class": validation_loader.__class__.__name__,
        "component_center_fraction": training_loader.lesion_uniform_center_fraction,
        "training_probability_sum": float(probabilities.sum()),
        "training_probability_minimum": float(probabilities.min()),
        "training_probability_maximum": float(probabilities.max()),
        "batch_keys": list(batch["keys"]),
        "data_shape": list(batch["data"].shape),
        "segmentation_shape": list(batch["target"].shape),
        "batch_foreground_voxels": int((batch["target"] == 1).sum()),
        "trainer_source": {
            "path": str(args.trainer_source.resolve()),
            "sha256": sha256(args.trainer_source),
        },
        "training_data_read": True,
        "training_data_modified": False,
        "gpu_used": False,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
