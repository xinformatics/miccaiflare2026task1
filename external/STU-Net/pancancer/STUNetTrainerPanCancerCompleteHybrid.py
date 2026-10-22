"""Completeness-aware STU-Net fallback for full-volume lesion recovery.

The challenge training labels are partial by default. Seven externally
documented cohorts, however, have exhaustive task-specific masks. This trainer
uses full-patch supervision only for those cohorts and retains component-box
ignore supervision for every other case. It is deliberately isolated from the
active balanced continuation.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence

import numpy as np
import torch

from STUNetTrainerPanCancer import (
    IGNORE_LABEL,
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
    component_box_target,
)
from STUNetTrainerPanCancerSourceBalanced import (
    identifier_inventory_sha256,
    unwrap_augmenter_loader,
)


COMPLETE_CASE_MULTIPLIER = 4.0
COMPLETE_IDENTIFIER_PREFIXES = (
    "HeadNeck_SegRap25_",
    "Liver-Colorectal-Metastases_",
    "Liver-Totalseg-lesions_",
    "WholeBody_LongitudinalCTLesion_",
    "WholeBody_MSWAL_",
    "WholeBody_autoPETCT_",
    "Chest_LNDb_",
)


def is_complete_identifier(identifier: str) -> bool:
    return str(identifier).startswith(COMPLETE_IDENTIFIER_PREFIXES)


def complete_priority_probabilities(
    identifiers: Sequence[str],
    complete_multiplier: float = COMPLETE_CASE_MULTIPLIER,
) -> np.ndarray:
    """Return normalized bounded sampling weights for verified-complete cases."""

    identifiers = tuple(str(identifier) for identifier in identifiers)
    if not identifiers:
        raise ValueError("training dataset has no identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training identifiers are not unique")
    if complete_multiplier < 1.0 or not np.isfinite(complete_multiplier):
        raise ValueError("complete_multiplier must be finite and at least 1")
    complete = np.asarray(
        [is_complete_identifier(identifier) for identifier in identifiers],
        dtype=bool,
    )
    if not complete.any() or complete.all():
        raise ValueError("training inventory must contain complete and partial cases")
    weights = np.where(complete, complete_multiplier, 1.0).astype(np.float64)
    probabilities = weights / weights.sum()
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("complete-priority probabilities are not normalized")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise AssertionError("sampling probabilities must be finite and positive")
    return probabilities


def complete_hybrid_target(
    target: torch.Tensor,
    keys: Sequence[str],
    full_shape: tuple[int, int, int],
    full_margin_voxels: tuple[int, int, int],
) -> torch.Tensor:
    """Use full binary masks for verified cohorts and boxes for partial cases."""

    if target.ndim != 5 or target.shape[1] != 1:
        raise ValueError("target must be [B,1,D,H,W]")
    keys = tuple(str(key) for key in keys)
    if len(keys) != target.shape[0]:
        raise ValueError("one case key is required for every target sample")
    result = torch.full_like(target, IGNORE_LABEL)
    for batch_index, key in enumerate(keys):
        source = target[batch_index : batch_index + 1]
        if not is_complete_identifier(key):
            result[batch_index : batch_index + 1] = component_box_target(
                source, full_shape, full_margin_voxels
            )
            continue
        known_binary = (source == 0) | (source == 1)
        result[batch_index : batch_index + 1] = torch.where(
            known_binary, source, torch.full_like(source, IGNORE_LABEL)
        )
    return result


def complete_priority_loader_factory(
    original_loader: Callable,
    complete_multiplier: float = COMPLETE_CASE_MULTIPLIER,
) -> Callable:
    """Inject complete-priority sampling into training, never validation."""

    call_index = 0

    def loader(*args, **kwargs):
        nonlocal call_index
        data = args[0] if args else kwargs.get("data")
        if data is None or not hasattr(data, "identifiers"):
            raise ValueError("nnU-Net loader data must expose identifiers")
        if call_index == 0:
            if kwargs.get("sampling_probabilities") is not None:
                raise ValueError("training sampling probabilities were already set")
            kwargs["sampling_probabilities"] = complete_priority_probabilities(
                data.identifiers, complete_multiplier
            )
        else:
            kwargs["sampling_probabilities"] = None
        call_index += 1
        return original_loader(*args, **kwargs)

    return loader


class STUNetTrainerPanCancerComponentBoxE160CompleteHybrid(
    STUNetTrainerPanCancerComponentBoxE160Balanced
):
    """Epoch-160 continuation with trusted complete-cohort background."""

    complete_case_multiplier = COMPLETE_CASE_MULTIPLIER
    expected_case_count = 17_575
    expected_complete_case_count = 3_145
    expected_partial_case_count = 14_430
    expected_identifier_sha256 = (
        "28f4d1bfd095422f546f0f53134eb4f226963243bc89fcdd78a6d7614bceb509"
    )

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        # Batch size is two: one position is random and one is guaranteed
        # foreground. Random partial-label crops safely contribute zero loss
        # when empty; random complete-label crops provide authoritative
        # lesion-free background supervision.
        self.oversample_foreground_percent = 0.5

    def _validate_identifiers(self, identifiers: Sequence[str]) -> None:
        identifiers = tuple(str(identifier) for identifier in identifiers)
        observed_sha = identifier_inventory_sha256(identifiers)
        if observed_sha != self.expected_identifier_sha256:
            raise ValueError(
                "complete-hybrid identifier inventory SHA256 mismatch: "
                f"{observed_sha}"
            )
        complete_count = sum(is_complete_identifier(value) for value in identifiers)
        partial_count = len(identifiers) - complete_count
        if len(identifiers) != self.expected_case_count:
            raise ValueError(
                f"expected {self.expected_case_count} cases, found {len(identifiers)}"
            )
        if complete_count != self.expected_complete_case_count:
            raise ValueError(
                "verified-complete case count mismatch: "
                f"{complete_count} != {self.expected_complete_case_count}"
            )
        if partial_count != self.expected_partial_case_count:
            raise ValueError(
                f"partial case count mismatch: {partial_count}"
            )

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_loader = trainer_module.nnUNetDataLoader
        trainer_module.nnUNetDataLoader = complete_priority_loader_factory(
            original_loader, self.complete_case_multiplier
        )
        try:
            loaders = super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader = original_loader
        training_loader = unwrap_augmenter_loader(loaders[0])
        self._validate_identifiers(training_loader._data.identifiers)
        return loaders

    def _hybrid_targets(self, target, keys: Sequence[str]):
        targets = target if isinstance(target, list) else [target]
        full_shape = tuple(int(value) for value in targets[0].shape[2:])
        converted = [
            complete_hybrid_target(item, keys, full_shape, self.full_margin_voxels)
            for item in targets
        ]
        return converted if isinstance(target, list) else converted[0]

    def _partial_targets(self, target):
        keys = getattr(self, "_active_batch_keys", None)
        if keys is None:
            raise RuntimeError("complete-hybrid target conversion requires batch keys")
        return self._hybrid_targets(target, keys)

    def _run_keyed_step(self, method: Callable, batch: dict) -> dict:
        if getattr(self, "_active_batch_keys", None) is not None:
            raise RuntimeError("nested complete-hybrid batch conversion")
        self._active_batch_keys = tuple(str(key) for key in batch["keys"])
        try:
            return method(self, batch)
        finally:
            self._active_batch_keys = None

    def train_step(self, batch: dict) -> dict:
        return self._run_keyed_step(STUNetTrainerPanCancer.train_step, batch)

    def validation_step(self, batch: dict) -> dict:
        return self._run_keyed_step(STUNetTrainerPanCancer.validation_step, batch)

    def on_train_start(self):
        STUNetTrainerPanCancer.on_train_start(self)
        if self.local_rank == 0:
            complete_mass = (
                self.expected_complete_case_count * self.complete_case_multiplier
                / (
                    self.expected_complete_case_count * self.complete_case_multiplier
                    + self.expected_partial_case_count
                )
            )
            self._initialize_wandb(
                name="stunetb_componentbox_e160_completehybrid4x_80ep_20260826",
                run_id="stunetbcompe160completehybrid4x20260826",
                partial_label_policy=(
                    "full_patch_for_verified_complete_else_component_box_ignore"
                ),
                loss_name="hybrid_complete_or_component_box_CE_plus_Tversky_a05_b05",
                extra_config={
                    "parent_epoch": 160,
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "optimizer_restart": True,
                    "optimizer": "SGD_Nesterov_momentum_0.99",
                    "lr_schedule": "polynomial_80_epochs",
                    "complete_case_multiplier": self.complete_case_multiplier,
                    "complete_case_probability_mass": complete_mass,
                    "oversample_foreground_percent": self.oversample_foreground_percent,
                    "expected_training_cases": self.expected_case_count,
                    "expected_complete_cases": self.expected_complete_case_count,
                    "expected_partial_cases": self.expected_partial_case_count,
                    "expected_identifier_sha256": self.expected_identifier_sha256,
                    "complete_identifier_prefixes": COMPLETE_IDENTIFIER_PREFIXES,
                },
            )
