#!/usr/bin/env python3
"""Workspace-local nnU-Net entry point for the FLARE25 multi-teacher student."""

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import nnunetv2.run.run_training as run_training_module

from nnunet.flare25_multiteacher_distill import (
    nnUNetTrainerFLARE25MultiTeacherDistill,
    nnUNetTrainerFLARE25MultiTeacherSmoke,
)


TRAINER_NAME = "nnUNetTrainerFLARE25MultiTeacherDistill"
SMOKE_TRAINER_NAME = "nnUNetTrainerFLARE25MultiTeacherSmoke"
_ORIGINAL_RESOLVER = run_training_module.recursive_find_python_class


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return nnUNetTrainerFLARE25MultiTeacherDistill
    if class_name == SMOKE_TRAINER_NAME:
        return nnUNetTrainerFLARE25MultiTeacherSmoke
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


def load_same_task_weights(network, filename, verbose=False):
    del verbose
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    pretrained = checkpoint["network_weights"]
    module = network.module if isinstance(network, DDP) else network
    module = getattr(module, "_orig_mod", module)
    current = module.state_dict()
    if set(current) != set(pretrained):
        raise RuntimeError("official FLARE25 checkpoint tensor names changed")
    mismatched = [key for key in current if current[key].shape != pretrained[key].shape]
    if mismatched:
        raise RuntimeError(f"official FLARE25 tensor shapes changed: {mismatched[:5]}")
    module.load_state_dict(pretrained, strict=True)
    print(f"Loaded all official FLARE25 tensors from {filename}", flush=True)


run_training_module.recursive_find_python_class = resolve_trainer
run_training_module.load_pretrained_weights = load_same_task_weights


if __name__ == "__main__":
    run_training_module.run_training_entry()
