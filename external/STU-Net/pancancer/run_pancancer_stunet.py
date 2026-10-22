#!/usr/bin/env python3
"""Launch STU-Net through nnU-Net 2.6 without modifying site-packages."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import numpy.core.multiarray as numpy_multiarray
import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from torch.serialization import safe_globals


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancer import (  # noqa: E402
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBox,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
    STUNetTrainerPanCancerSmoke,
)

import nnunetv2.run.run_training as run_training_module  # noqa: E402


EXPECTED_PRETRAINED_SHA256 = (
    "7232a6296d320da1d60debe873a1d3fcad74e9404272db8f7440e30d62c5997a"
)
EXPECTED_COMPONENT_E160_SHA256 = (
    "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"
)
TRAINER_NAME = "STUNetTrainerPanCancer"
COMPONENT_BOX_TRAINER_NAME = "STUNetTrainerPanCancerComponentBox"
COMPONENT_BOX_E160_BALANCED_TRAINER_NAME = (
    "STUNetTrainerPanCancerComponentBoxE160Balanced"
)
SMOKE_TRAINER_NAME = "STUNetTrainerPanCancerSmoke"
_ORIGINAL_RESOLVER = run_training_module.recursive_find_python_class


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return STUNetTrainerPanCancer
    if class_name == COMPONENT_BOX_TRAINER_NAME:
        return STUNetTrainerPanCancerComponentBox
    if class_name == COMPONENT_BOX_E160_BALANCED_TRAINER_NAME:
        return STUNetTrainerPanCancerComponentBoxE160Balanced
    if class_name == SMOKE_TRAINER_NAME:
        return STUNetTrainerPanCancerSmoke
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


def load_stunet_pretrained_weights(network, filename, verbose=False):
    path = Path(filename)
    observed = sha256_file(path)
    if observed not in {
        EXPECTED_PRETRAINED_SHA256,
        EXPECTED_COMPONENT_E160_SHA256,
    }:
        raise ValueError(f"STU-Net-B checkpoint SHA256 mismatch: {observed}")
    allowed = [
        (numpy_multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]
    with safe_globals(allowed):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    model = network.module if isinstance(network, DistributedDataParallel) else network
    if isinstance(model, OptimizedModule):
        model = model._orig_mod
    current = model.state_dict()

    if observed == EXPECTED_PRETRAINED_SHA256:
        pretrained = checkpoint.get("state_dict")
        if (
            not isinstance(pretrained, dict)
            or int(checkpoint.get("epoch", -1)) != 4000
        ):
            raise ValueError("invalid STU-Net-B pretrained checkpoint contract")
        for key, value in current.items():
            if "seg_outputs" in key:
                continue
            if key not in pretrained or pretrained[key].shape != value.shape:
                raise ValueError(f"incompatible STU-Net-B non-head tensor: {key}")
        transferable = {
            key: value
            for key, value in pretrained.items()
            if key in current and "seg_outputs" not in key
        }
        source_description = "verified STU-Net-B epoch 4000 checkpoint"
    else:
        pretrained = checkpoint.get("network_weights")
        if (
            not isinstance(pretrained, dict)
            or checkpoint.get("checkpoint_kind") != "inference_only"
            or int(checkpoint.get("current_epoch", -1)) != 160
            or checkpoint.get("trainer_name")
            != "STUNetTrainerPanCancerComponentBox"
        ):
            raise ValueError("invalid component-box epoch-160 parent contract")
        if set(pretrained) != set(current):
            raise ValueError("component-box epoch-160 tensor inventory mismatch")
        for key, value in current.items():
            if pretrained[key].shape != value.shape:
                raise ValueError(f"incompatible component epoch-160 tensor: {key}")
        transferable = pretrained
        source_description = "verified component-box epoch-160 checkpoint"

    current.update(transferable)
    model.load_state_dict(current, strict=True)
    print(
        f"Loaded {len(transferable)} tensors from {source_description} ({observed})",
        flush=True,
    )
    if verbose and observed == EXPECTED_PRETRAINED_SHA256:
        print("Segmentation heads were intentionally reinitialized for binary lesions.")


# These assignments occur in every multiprocessing-spawned child before the
# nnU-Net DDP entry point executes.
run_training_module.recursive_find_python_class = resolve_trainer
run_training_module.load_pretrained_weights = load_stunet_pretrained_weights


if __name__ == "__main__":
    run_training_module.run_training_entry()
