"""Post-base ResEnc-S6 tuning with positive-only Train-Unlabel supervision."""

from __future__ import annotations

import importlib
import os
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.distributed as dist
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from threadpoolctl import threadpool_limits

# Workspace roots are injected by nnunet.resenc_s6_partialbox before importing
# its STU/GF dependencies.
from nnunet.resenc_s6_partialbox import nnUNetTrainerResEncS6PanCancerPartialBox
from STUNetTrainerPanCancer import STUNetTrainerPanCancer
from STUNetTrainerPanCancerSourceBalanced import (
    component_uniform_voxel,
    foreground_proxy_counts_from_dataset,
    identifier_inventory_sha256,
    source_from_identifier,
    stratified_source_probabilities,
)
from fault_tolerant_wandb import safe_wandb_init, stable_wandb_run_id


PSEUDO_PREFIX = "Pseudo-"
PSEUDO_EXPOSURE = 0.20
EXPECTED_LABELED_CASES = 17_575
EXPECTED_PSEUDO_CASES = 2_311
EXPECTED_TOTAL_CASES = EXPECTED_LABELED_CASES + EXPECTED_PSEUDO_CASES
EXPECTED_PSEUDO_COHORTS = {"Pseudo-AMOS-2350": 1_843, "Pseudo-MSD-506": 468}
EXPECTED_SORTED_IDENTIFIER_SHA256 = (
    "37e3c103543fa4c737694be9a44b80c02aae6c6aa2161622e29bd517a4af728e"
)


def positive_unlabeled_sampling_probabilities(
    identifiers: Sequence[str],
    foreground_counts: Sequence[int],
    pseudo_exposure: float = PSEUDO_EXPOSURE,
) -> np.ndarray:
    """Reserve fixed pseudo exposure; source-balance only labeled cases."""

    identifiers = tuple(str(identifier) for identifier in identifiers)
    foreground_counts = np.asarray(foreground_counts, dtype=np.int64)
    if foreground_counts.shape != (len(identifiers),):
        raise ValueError("foreground counts must align with identifiers")
    if not 0.0 < pseudo_exposure < 1.0:
        raise ValueError("pseudo exposure must lie strictly between zero and one")
    pseudo = np.asarray(
        [identifier.startswith(PSEUDO_PREFIX) for identifier in identifiers],
        dtype=bool,
    )
    if not pseudo.any() or pseudo.all():
        raise ValueError("combined training requires labeled and pseudo cases")
    labeled_indices = np.flatnonzero(~pseudo)
    pseudo_indices = np.flatnonzero(pseudo)
    labeled_probabilities = stratified_source_probabilities(
        [identifiers[index] for index in labeled_indices],
        foreground_counts[labeled_indices],
    )
    result = np.zeros(len(identifiers), dtype=np.float64)
    result[labeled_indices] = (1.0 - pseudo_exposure) * labeled_probabilities
    result[pseudo_indices] = pseudo_exposure / len(pseudo_indices)
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("combined sampling probabilities do not sum to one")
    if not np.isclose(result[pseudo].sum(), pseudo_exposure, rtol=0.0, atol=1e-12):
        raise AssertionError("pseudo exposure changed")
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise AssertionError("combined sampling probabilities must be positive")
    return result


def positive_unlabeled_loader_factory(
    original_loader: type,
    component_center_fraction: float,
    foreground_count_resolver: Callable,
    pseudo_exposure: float = PSEUDO_EXPOSURE,
) -> Callable:
    """Inject controlled exposure and component-uniform training centers."""

    if not isinstance(original_loader, type):
        raise TypeError("positive-unlabeled loader requires an nnU-Net loader class")
    call_index = 0

    class PositiveUnlabeledLoader(original_loader):
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
            kwargs["sampling_probabilities"] = positive_unlabeled_sampling_probabilities(
                data.identifiers, counts, pseudo_exposure
            )
            result = PositiveUnlabeledLoader(*args, **kwargs)
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


class nnUNetTrainerResEncS6PanCancerPositiveUnlabeled(
    nnUNetTrainerResEncS6PanCancerPartialBox
):
    """Fine-tune the efficient base S6 with positive-only pseudo supervision."""

    landmark_epochs = (25, 50)
    positive_stage_end_epochs = (25, 50, 100)
    component_center_fraction = 0.25

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.initial_lr = 1e-4
        stage_end_epoch = int(os.environ.get("RESENC_POSITIVE_STAGE_END_EPOCH", "100"))
        if stage_end_epoch not in self.positive_stage_end_epochs:
            raise ValueError(
                "RESENC_POSITIVE_STAGE_END_EPOCH must be one of "
                f"{self.positive_stage_end_epochs}, found {stage_end_epoch}"
            )
        self.num_epochs = stage_end_epoch
        self.lr_schedule_horizon_epochs = 100
        self.save_every = 1
        self.inference_checkpoint_start_epoch = 101

    def _validated_counts(self, data) -> np.ndarray:
        identifiers = tuple(data.identifiers)
        if len(identifiers) != EXPECTED_TOTAL_CASES:
            raise ValueError("positive-unlabeled case count changed")
        observed_hash = identifier_inventory_sha256(sorted(identifiers))
        if observed_hash != EXPECTED_SORTED_IDENTIFIER_SHA256:
            raise ValueError(f"positive-unlabeled identifier SHA256 changed: {observed_hash}")
        pseudo = [identifier for identifier in identifiers if identifier.startswith(PSEUDO_PREFIX)]
        labeled = [identifier for identifier in identifiers if not identifier.startswith(PSEUDO_PREFIX)]
        if len(labeled) != EXPECTED_LABELED_CASES or len(pseudo) != EXPECTED_PSEUDO_CASES:
            raise ValueError("positive-unlabeled partition changed")
        cohorts = Counter(source_from_identifier(identifier) for identifier in pseudo)
        if dict(cohorts) != EXPECTED_PSEUDO_COHORTS:
            raise ValueError(f"positive-unlabeled cohort counts changed: {cohorts}")
        return foreground_proxy_counts_from_dataset(data)

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_loader = trainer_module.nnUNetDataLoader
        trainer_module.nnUNetDataLoader = positive_unlabeled_loader_factory(
            original_loader,
            self.component_center_fraction,
            self._validated_counts,
        )
        try:
            loaders = STUNetTrainerPanCancer.get_dataloaders(self)
        finally:
            trainer_module.nnUNetDataLoader = original_loader
        training_loader = unwrap_loader(loaders[0])
        probabilities = np.asarray(training_loader.sampling_probabilities)
        identifiers = tuple(training_loader._data.identifiers)
        pseudo = np.asarray([item.startswith(PSEUDO_PREFIX) for item in identifiers])
        if (
            probabilities.shape != (EXPECTED_TOTAL_CASES,)
            or not np.isclose(probabilities[pseudo].sum(), PSEUDO_EXPOSURE, rtol=0.0, atol=1e-12)
            or getattr(training_loader, "lesion_uniform_center_fraction", None) != 0.25
        ):
            raise ValueError("positive-unlabeled loader contract mismatch")
        return loaders

    def on_train_start(self) -> None:
        nnUNetTrainer.on_train_start(self)
        if not self.is_ddp or dist.get_world_size() != 2:
            raise RuntimeError("positive-unlabeled S6 training requires exactly 2 GPUs")
        if self.configuration_manager.batch_size != self.expected_global_batch_size:
            raise RuntimeError("positive-unlabeled S6 batch-size contract changed")
        parameter_source = self.network.module if self.is_ddp else self.network
        parameter_source = getattr(parameter_source, "_orig_mod", parameter_source)
        parameter_count = sum(parameter.numel() for parameter in parameter_source.parameters())
        if parameter_count != self.expected_parameter_count:
            raise RuntimeError("positive-unlabeled S6 parameter count changed")
        if self.local_rank != 0:
            return
        run_name = "resenc_s6_positive_unlabeled_f020_100ep_20260827"
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
                "initialization": "base_ResEncS6_epoch500_network_weights",
                "global_batch_size": self.expected_global_batch_size,
                "initial_lr": self.initial_lr,
                "num_epochs": self.num_epochs,
                "optimizer": "SGD_Nesterov_momentum_0.99_fresh_state",
                "lr_schedule": "polynomial_100_epochs",
                "loss": "component-box_CE_plus_Tversky_a025_b075",
                "outside_boxes": "ignore_for_labeled_and_pseudo",
                "labeled_cases": EXPECTED_LABELED_CASES,
                "positive_unlabeled_cases": EXPECTED_PSEUDO_CASES,
                "pseudo_exposure": PSEUDO_EXPOSURE,
                "empty_unlabeled_cases": "excluded_unknown_not_negative",
                "public_used_for_training": False,
                "hidden_used_for_training": False,
            },
        )
