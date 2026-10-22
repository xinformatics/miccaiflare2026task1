#!/usr/bin/env python3
"""Launch the isolated shape-following partial-band ResEnc-S6 trainer."""

from __future__ import annotations

import nnunetv2.run.run_training as run_training_module

from nnunet.resenc_s6_partialband12 import (
    nnUNetTrainerResEncS6PanCancerPartialBand12,
)


TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerPartialBand12"
_ORIGINAL_RESOLVER = run_training_module.recursive_find_python_class


def resolve_trainer(folder, class_name, current_module):
    if class_name == TRAINER_NAME:
        return nnUNetTrainerResEncS6PanCancerPartialBand12
    return _ORIGINAL_RESOLVER(folder, class_name, current_module)


run_training_module.recursive_find_python_class = resolve_trainer


if __name__ == "__main__":
    run_training_module.run_training_entry()
