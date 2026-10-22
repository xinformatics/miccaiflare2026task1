"""Cautious organ-target continuation for stomach/bladder lesion recall."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from threadpoolctl import threadpool_limits

from nnunet.resenc_s6_partialbox import nnUNetTrainerResEncS6PanCancerPartialBox
from STUNetTrainerPanCancer import STUNetTrainerPanCancer
from STUNetTrainerPanCancerSourceBalanced import (
    component_uniform_voxel,
    foreground_proxy_counts_from_dataset,
    identifier_inventory_sha256,
    stratified_source_probabilities,
)
from fault_tolerant_wandb import safe_wandb_init, stable_wandb_run_id


ORGAN_PREFIX = "PseudoOrgan-"
ORGAN_TARGET_EXPOSURE = 0.01
EXPECTED_LABELED_CASES = 17_575
EXPECTED_ORGAN_TARGET_CASES = 4
EXPECTED_TOTAL_CASES = 17_579
EXPECTED_SORTED_IDENTIFIER_SHA256 = (
    "1df4a798cb3acf4b674d90f5ea1120cdb0fb08fb481319c289f339daa57d0991"
)


def organ_target_sampling_probabilities(
    identifiers: Sequence[str],
    foreground_counts: Sequence[int],
    target_exposure: float = ORGAN_TARGET_EXPOSURE,
) -> np.ndarray:
    """Reserve one percent for four focal targets; preserve labeled mixture."""

    identifiers = tuple(str(identifier) for identifier in identifiers)
    foreground_counts = np.asarray(foreground_counts, dtype=np.int64)
    if foreground_counts.shape != (len(identifiers),):
        raise ValueError("foreground counts must align with identifiers")
    if not 0.0 < target_exposure < 1.0:
        raise ValueError("organ-target exposure must lie strictly between zero and one")
    target = np.asarray(
        [identifier.startswith(ORGAN_PREFIX) for identifier in identifiers], dtype=bool
    )
    if int(target.sum()) != EXPECTED_ORGAN_TARGET_CASES or target.all():
        raise ValueError("combined training requires exactly four organ-target cases")
    labeled_indices = np.flatnonzero(~target)
    target_indices = np.flatnonzero(target)
    labeled_probabilities = stratified_source_probabilities(
        [identifiers[index] for index in labeled_indices],
        foreground_counts[labeled_indices],
    )
    result = np.zeros(len(identifiers), dtype=np.float64)
    result[labeled_indices] = (1.0 - target_exposure) * labeled_probabilities
    result[target_indices] = target_exposure / len(target_indices)
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("organ-target probabilities do not sum to one")
    if not np.isclose(result[target].sum(), target_exposure, rtol=0.0, atol=1e-12):
        raise AssertionError("organ-target exposure changed")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise AssertionError("organ-target probabilities must be finite and positive")
    return result


def organ_target_loader_factory(
    original_loader: type,
    component_center_fraction: float,
    foreground_count_resolver: Callable,
) -> Callable:
    """Inject training-only target exposure and component-uniform centers."""

    if not isinstance(original_loader, type):
        raise TypeError("organ-target loader requires an nnU-Net loader class")
    call_index = 0

    class OrganTargetLoader(original_loader):
        lesion_uniform_center_fraction = float(component_center_fraction)

        def generate_train_batch(self):
            selected_keys = self.get_indices()
            data_all = np.zeros(self.data_shape, dtype=np.float32)
            seg_all = np.zeros(self.seg_shape, dtype=np.int16)
            for batch_index, identifier in enumerate(selected_keys):
                force_fg = self.get_do_oversample(batch_index)
                data, seg, seg_prev, properties = self._data.load_case(identifier)
                shape = data.shape[1:]
                selected_voxel = (
                    component_uniform_voxel(
                        seg, self.lesion_uniform_center_fraction, np.random
                    )
                    if force_fg
                    else None
                )
                class_locations = (
                    properties["class_locations"]
                    if selected_voxel is None
                    else {1: selected_voxel[None]}
                )
                bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)
                bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]
                data_all[batch_index] = crop_and_pad_nd(data, bbox, 0)
                seg_cropped = crop_and_pad_nd(seg, bbox, -1)
                if seg_prev is not None:
                    seg_cropped = np.vstack(
                        (seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None])
                    )
                seg_all[batch_index] = seg_cropped
            if self.patch_size_was_2d:
                data_all = data_all[:, :, 0]
                seg_all = seg_all[:, :, 0]
            if self.transforms is None:
                return {"data": data_all, "target": seg_all, "keys": selected_keys}
            with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
                data_tensor = torch.from_numpy(data_all).float()
                seg_tensor = torch.from_numpy(seg_all).to(torch.int16)
                images, segmentations = [], []
                for batch_index in range(self.batch_size):
                    transformed = self.transforms(
                        image=data_tensor[batch_index],
                        segmentation=seg_tensor[batch_index],
                    )
                    images.append(transformed["image"])
                    segmentations.append(transformed["segmentation"])
                data_result = torch.stack(images)
                if isinstance(segmentations[0], list):
                    seg_result = [
                        torch.stack([item[index] for item in segmentations])
                        for index in range(len(segmentations[0]))
                    ]
                else:
                    seg_result = torch.stack(segmentations)
            return {"data": data_result, "target": seg_result, "keys": selected_keys}

    def loader(*args, **kwargs):
        nonlocal call_index
        data = args[0] if args else kwargs.get("data")
        if data is None or not hasattr(data, "identifiers"):
            raise ValueError("nnU-Net loader data must expose identifiers")
        if call_index == 0:
            if kwargs.get("sampling_probabilities") is not None:
                raise ValueError("training sampling probabilities were already set")
            counts = foreground_count_resolver(data)
            kwargs["sampling_probabilities"] = organ_target_sampling_probabilities(
                data.identifiers, counts
            )
            result = OrganTargetLoader(*args, **kwargs)
        else:
            kwargs["sampling_probabilities"] = None
            result = original_loader(*args, **kwargs)
        call_index += 1
        return result

    return loader


def unwrap_loader(augmenter):
    loader = getattr(augmenter, "data_loader", None)
    if loader is None:
        loader = getattr(augmenter, "generator", None)
    if loader is None or not hasattr(loader, "_data"):
        raise ValueError("unable to resolve nnU-Net training loader")
    return loader


class nnUNetTrainerResEncS6PanCancerOrganTarget(
    nnUNetTrainerResEncS6PanCancerPartialBox
):
    """Fine-tune a positive-unlabeled parent on sparse organ-target evidence."""

    landmark_epochs = (10,)
    organ_stage_end_epochs = (10, 25)
    component_center_fraction = 0.25

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 3e-5
        stage_end_epoch = int(os.environ.get("RESENC_ORGAN_STAGE_END_EPOCH", "25"))
        if stage_end_epoch not in self.organ_stage_end_epochs:
            raise ValueError(
                "RESENC_ORGAN_STAGE_END_EPOCH must be one of "
                f"{self.organ_stage_end_epochs}, found {stage_end_epoch}"
            )
        self.num_epochs = stage_end_epoch
        self.lr_schedule_horizon_epochs = 25
        self.save_every = 1
        self.inference_checkpoint_start_epoch = 26

    def _validated_counts(self, data) -> np.ndarray:
        identifiers = tuple(data.identifiers)
        if len(identifiers) != EXPECTED_TOTAL_CASES:
            raise ValueError("organ-target case count changed")
        observed_hash = identifier_inventory_sha256(sorted(identifiers))
        if observed_hash != EXPECTED_SORTED_IDENTIFIER_SHA256:
            raise ValueError(f"organ-target identifier SHA256 changed: {observed_hash}")
        target = [item for item in identifiers if item.startswith(ORGAN_PREFIX)]
        if len(target) != EXPECTED_ORGAN_TARGET_CASES:
            raise ValueError("organ-target partition changed")
        return foreground_proxy_counts_from_dataset(data)

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_loader = trainer_module.nnUNetDataLoader
        trainer_module.nnUNetDataLoader = organ_target_loader_factory(
            original_loader, self.component_center_fraction, self._validated_counts
        )
        try:
            loaders = STUNetTrainerPanCancer.get_dataloaders(self)
        finally:
            trainer_module.nnUNetDataLoader = original_loader
        training_loader = unwrap_loader(loaders[0])
        probabilities = np.asarray(training_loader.sampling_probabilities)
        identifiers = tuple(training_loader._data.identifiers)
        target = np.asarray(
            [item.startswith(ORGAN_PREFIX) for item in identifiers], dtype=bool
        )
        if (
            probabilities.shape != (EXPECTED_TOTAL_CASES,)
            or not np.isclose(
                probabilities[target].sum(), ORGAN_TARGET_EXPOSURE, rtol=0.0, atol=1e-12
            )
            or getattr(training_loader, "lesion_uniform_center_fraction", None) != 0.25
        ):
            raise ValueError("organ-target loader contract mismatch")
        return loaders

    def on_train_start(self) -> None:
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("organ-target S6 training requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("organ-target S6 batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError("organ-target S6 parameter count changed")
        if self.local_rank != 0:
            return
        run_name = "resenc_s6_organ_target_f001_25ep_20260827"
        run_id = stable_wandb_run_id(self.output_folder, run_name)
        self._resenc_wandb_run = safe_wandb_init(
            project=os.environ.get("WANDB_PROJECT", "pancancer-ct"),
            name=run_name,
            id=run_id,
            resume="allow",
            mode=os.environ.get("WANDB_MODE", "online"),
            config={
                "architecture": self.architecture_name,
                "parameters": self.expected_parameter_count,
                "initialization": "frozen_positive_public_selected_or_diagnostic_checkpoint",
                "global_batch_size": self.expected_global_batch_size,
                "initial_lr": self.initial_lr,
                "num_epochs": self.num_epochs,
                "optimizer": "SGD_Nesterov_momentum_0.99_fresh_state",
                "lr_schedule": "polynomial_25_epochs",
                "loss": "component-box_CE_plus_Tversky_a025_b075",
                "outside_boxes": "ignore_for_labeled_and_pseudo",
                "labeled_cases": EXPECTED_LABELED_CASES,
                "organ_target_cases": EXPECTED_ORGAN_TARGET_CASES,
                "organ_target_exposure": ORGAN_TARGET_EXPOSURE,
                "target_organs": ["stomach", "urinary_bladder"],
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )
