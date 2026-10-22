#!/usr/bin/env python3
"""Launch FLARE25 partial-box fine-tuning with exact same-task weights."""

from __future__ import annotations

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import nnunetv2.run.run_training as run_training_module

from nnunet.flare25_partialbox_finetune import (
    nnUNetTrainerFLARE25PartialBoxFineTune,
)


TRAINER_NAME = "nnUNetTrainerFLARE25PartialBoxFineTune"
_ORIGINAL_RESOLVER = run_training_module.recursive_find_python_class


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return nnUNetTrainerFLARE25PartialBoxFineTune
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


def load_same_task_weights(network, filename, verbose=False):
    """Load every released tensor, including the binary segmentation heads."""

    del verbose
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    pretrained = checkpoint["network_weights"]
    module = network.module if isinstance(network, DDP) else network
    module = getattr(module, "_orig_mod", module)
    current = module.state_dict()
    missing = sorted(set(current) - set(pretrained))
    extra = sorted(set(pretrained) - set(current))
    mismatched = sorted(
        key
        for key in set(current) & set(pretrained)
        if current[key].shape != pretrained[key].shape
    )
    if missing or extra or mismatched:
        raise RuntimeError(
            "FLARE25 checkpoint is not an exact architecture match: "
            f"missing={missing[:5]}, extra={extra[:5]}, "
            f"shape_mismatch={mismatched[:5]}"
        )
    module.load_state_dict(pretrained, strict=True)
    print(
        "Loaded all official FLARE25 network tensors, including segmentation "
        f"heads, from {filename}"
    )


run_training_module.recursive_find_python_class = resolve_trainer
run_training_module.load_pretrained_weights = load_same_task_weights


if __name__ == "__main__":
    run_training_module.run_training_entry()

