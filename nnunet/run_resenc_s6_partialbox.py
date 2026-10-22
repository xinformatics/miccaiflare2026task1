#!/usr/bin/env python3
"""Launch the workspace-local partial-label-safe ResEnc S6 trainer."""

from __future__ import annotations

import nnunetv2.run.run_training as run_training_module

from nnunet.resenc_s6_partialbox import (
    nnUNetTrainerResEncS6PanCancerPartialBox,
)


TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerPartialBox"
_ORIGINAL_RESOLVER = run_training_module.recursive_find_python_class


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return nnUNetTrainerResEncS6PanCancerPartialBox
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


run_training_module.recursive_find_python_class = resolve_trainer


if __name__ == "__main__":
    run_training_module.run_training_entry()

