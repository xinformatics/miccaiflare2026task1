"""Make the released ULS+ identity resampler visible to spawned nnU-Net workers.

Python imports ``sitecustomize`` automatically at interpreter startup when
this directory is on ``PYTHONPATH``.  nnU-Net preprocessing uses spawn, so a
parent-only monkey patch is insufficient.  Keeping this shim beside the model
avoids modifying the shared site-packages installation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_identity_resampler():
    source = (
        Path(__file__).resolve().parents[1]
        / "nnunetv2/preprocessing/resampling/custom_resampling.py"
    )
    spec = importlib.util.spec_from_file_location("uls_plus_custom_resampling", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ULS+ identity resampler from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.no_resampling_data_or_seg_to_shape


identity_resampler = _load_identity_resampler()

import nnunetv2.preprocessing.resampling.utils as resampling_utils
import nnunetv2.utilities.plans_handling.plans_handler as plans_handler


if not getattr(resampling_utils, "_uls_plus_patched", False):
    _original = resampling_utils.recursive_find_resampling_fn_by_name

    def _resolve(name: str):
        if name == "no_resampling_data_or_seg_to_shape":
            return identity_resampler
        return _original(name)

    resampling_utils.recursive_find_resampling_fn_by_name = _resolve
    plans_handler.recursive_find_resampling_fn_by_name = _resolve
    resampling_utils._uls_plus_patched = True
