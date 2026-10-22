"""Size-preserving, recall-biased source-stratified STU-Net continuation.

Architecture, epoch-160 parent weights, component-box supervision, optimizer,
learning rate, augmentations, batch size, and schedule remain fixed. Sampling
is the predeclared bounded source/size mixture. The loss is made mildly more
recall-aggressive than the epoch-160 parent (Tversky alpha=0.25, beta=0.75)
after the balanced alpha=0.5, beta=0.5 continuation contracted public
proposals at its epoch-40 endpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import pickle
from collections.abc import Callable, Sequence
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from scipy import ndimage
from threadpoolctl import threadpool_limits

from STUNetTrainerPanCancer import (
    PartialBoxTverskyCELoss,
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
)


SOURCE_FRACTION = 0.25
RAW_RATIO_CAP = 3.0
COMPONENT_CENTER_FRACTION = 0.25
SIZE_BIN_UPPER_BOUNDS = (100, 500, 1_000, 3_000, 9_999)


def source_from_identifier(identifier: str) -> str:
    if "_" not in identifier:
        raise ValueError(f"identifier has no source separator: {identifier!r}")
    source, case_token = identifier.rsplit("_", 1)
    if not source or not case_token:
        raise ValueError(f"invalid identifier: {identifier!r}")
    return source


def identifier_inventory_sha256(identifiers: Sequence[str]) -> str:
    payload = ("\n".join(str(identifier) for identifier in identifiers) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def foreground_proxy_count(properties: dict) -> int:
    """Return nnU-Net's stored foreground-location count (capped at 10,000)."""

    for key, value in properties.get("class_locations", {}).items():
        if str(key) in {"1", "(1,)"}:
            return int(len(value))
    return 0


def foreground_proxy_counts_from_dataset(data) -> np.ndarray:
    if not hasattr(data, "source_folder"):
        raise ValueError("nnU-Net training dataset has no source_folder")
    folder = Path(data.source_folder)
    counts = []
    for identifier in data.identifiers:
        path = folder / f"{identifier}.pkl"
        if not path.is_file():
            raise ValueError(f"missing nnU-Net case properties: {path}")
        with path.open("rb") as handle:
            counts.append(foreground_proxy_count(pickle.load(handle)))
    return np.asarray(counts, dtype=np.int64)


def size_bin(foreground_count: int) -> int:
    for index, upper in enumerate(SIZE_BIN_UPPER_BOUNDS):
        if foreground_count <= upper:
            return index
    return len(SIZE_BIN_UPPER_BOUNDS)


def stratified_source_probabilities(
    identifiers: Sequence[str],
    foreground_counts: Sequence[int],
    source_fraction: float = SOURCE_FRACTION,
    raw_ratio_cap: float = RAW_RATIO_CAP,
) -> np.ndarray:
    """Balance sources within size bins while preserving every bin's mass."""

    identifiers = tuple(str(identifier) for identifier in identifiers)
    if not identifiers:
        raise ValueError("training dataset has no identifiers")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training identifiers are not unique")
    foreground_counts = np.asarray(foreground_counts, dtype=np.int64)
    if foreground_counts.shape != (len(identifiers),):
        raise ValueError("foreground_counts must align one-to-one with identifiers")
    if np.any(foreground_counts < 0):
        raise ValueError("foreground_counts must be nonnegative")
    if not 0.0 <= source_fraction <= 1.0:
        raise ValueError("source_fraction must lie in [0, 1]")
    if raw_ratio_cap < 1.0:
        raise ValueError("raw_ratio_cap must be at least 1")
    sources = tuple(source_from_identifier(identifier) for identifier in identifiers)
    bins = np.asarray([size_bin(int(count)) for count in foreground_counts])
    case_count = len(identifiers)
    bin_counts = Counter(int(value) for value in bins)
    cell_counts = Counter(
        (source, int(bin_index))
        for source, bin_index in zip(sources, bins, strict=True)
    )
    bin_sources = {
        bin_index: {
            source
            for source, candidate_bin in zip(sources, bins, strict=True)
            if candidate_bin == bin_index
        }
        for bin_index in bin_counts
    }
    raw_target = np.asarray(
        [
            (bin_counts[int(bin_index)] / case_count)
            / len(bin_sources[int(bin_index)])
            / cell_counts[(source, int(bin_index))]
            for source, bin_index in zip(sources, bins, strict=True)
        ],
        dtype=np.float64,
    )
    bounded_ratio = np.minimum(raw_target * case_count, raw_ratio_cap)
    bounded_target = np.empty(case_count, dtype=np.float64)
    for bin_index, count in bin_counts.items():
        mask = bins == bin_index
        bounded_target[mask] = (
            (count / case_count)
            * bounded_ratio[mask]
            / bounded_ratio[mask].sum()
        )
    probabilities = (
        (1.0 - source_fraction) / case_count
        + source_fraction * bounded_target
    )
    if not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError(
            f"source-stratified probabilities sum to {probabilities.sum()}"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise AssertionError("sampling probabilities must be finite and positive")
    uniform = np.full(case_count, 1.0 / case_count)
    for bin_index in bin_counts:
        mask = bins == bin_index
        if not np.isclose(
            probabilities[mask].sum(), uniform[mask].sum(), rtol=0.0, atol=1e-12
        ):
            raise AssertionError(f"size-bin {bin_index} probability mass changed")
    return probabilities


def source_balanced_loader_factory(
    original_loader: Callable,
    source_fraction: float = SOURCE_FRACTION,
    raw_ratio_cap: float = RAW_RATIO_CAP,
    foreground_count_resolver: Callable = foreground_proxy_counts_from_dataset,
) -> Callable:
    """Inject probabilities into the first (training) nnU-Net loader only."""

    call_index = 0

    def loader(*args, **kwargs):
        nonlocal call_index
        data = args[0] if args else kwargs.get("data")
        if data is None or not hasattr(data, "identifiers"):
            raise ValueError("nnU-Net loader data must expose identifiers")
        if call_index == 0:
            if kwargs.get("sampling_probabilities") is not None:
                raise ValueError("training sampling probabilities were already set")
            foreground_counts = foreground_count_resolver(data)
            kwargs["sampling_probabilities"] = stratified_source_probabilities(
                data.identifiers,
                foreground_counts,
                source_fraction,
                raw_ratio_cap,
            )
        else:
            # Validation must retain ordinary uniform-case sampling.
            kwargs["sampling_probabilities"] = None
        call_index += 1
        return original_loader(*args, **kwargs)

    return loader


def component_uniform_voxel(
    segmentation: np.ndarray,
    component_center_fraction: float,
    random_state=np.random,
) -> np.ndarray | None:
    """Select one annotated lesion uniformly, then one voxel within it.

    Returning ``None`` delegates to nnU-Net's ordinary foreground-voxel
    sampling. The selected coordinate includes the leading segmentation
    channel index, matching nnU-Net's ``class_locations`` representation.
    """

    if not 0.0 <= component_center_fraction <= 1.0:
        raise ValueError("component_center_fraction must be in [0,1]")
    if random_state.random() >= component_center_fraction:
        return None
    foreground = np.asarray(segmentation[0] == 1)
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        return None
    lower = np.asarray([int(axis.min()) for axis in coordinates], dtype=np.int64)
    region = tuple(
        slice(int(start), int(axis.max()) + 1)
        for start, axis in zip(lower, coordinates, strict=True)
    )
    labeled, count = ndimage.label(
        foreground[region], structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count < 1:
        return None
    selected_component = int(random_state.randint(1, count + 1))
    component_coordinates = np.argwhere(labeled == selected_component)
    selected = component_coordinates[
        int(random_state.randint(0, len(component_coordinates)))
    ].astype(np.int64) + lower
    return np.concatenate((np.asarray([0], dtype=np.int64), selected))


def component_center_source_balanced_loader_factory(
    original_loader: type,
    source_fraction: float,
    raw_ratio_cap: float,
    component_center_fraction: float,
    foreground_count_resolver: Callable,
) -> Callable:
    """Inject source probabilities and component-aware centers into training."""

    if not isinstance(original_loader, type):
        raise TypeError("component-center loader requires an nnU-Net loader class")
    call_index = 0

    class ComponentCenterLoader(original_loader):
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
                if selected_voxel is None:
                    class_locations = properties["class_locations"]
                else:
                    class_locations = {1: selected_voxel[None]}
                bbox_lbs, bbox_ubs = self.get_bbox(
                    shape, force_fg, class_locations
                )
                bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]
                data_all[batch_index] = crop_and_pad_nd(data, bbox, 0)
                seg_cropped = crop_and_pad_nd(seg, bbox, -1)
                if seg_prev is not None:
                    previous = crop_and_pad_nd(seg_prev, bbox, -1)[None]
                    seg_cropped = np.vstack((seg_cropped, previous))
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
            foreground_counts = foreground_count_resolver(data)
            kwargs["sampling_probabilities"] = stratified_source_probabilities(
                data.identifiers,
                foreground_counts,
                source_fraction,
                raw_ratio_cap,
            )
            result = ComponentCenterLoader(*args, **kwargs)
        else:
            kwargs["sampling_probabilities"] = None
            result = original_loader(*args, **kwargs)
        call_index += 1
        return result

    return loader


def unwrap_augmenter_loader(augmenter):
    """Return the underlying nnU-Net loader for either augmenter backend."""

    loader = getattr(augmenter, "data_loader", None)
    if loader is None:
        loader = getattr(augmenter, "generator", None)
    if loader is None or not hasattr(loader, "_data"):
        raise ValueError("unable to resolve the underlying nnU-Net data loader")
    return loader


class STUNetTrainerPanCancerComponentBoxE160SourceStratified(
    STUNetTrainerPanCancerComponentBoxE160Balanced
):
    """Mildly recall-aggressive continuation with bounded source sampling."""

    source_fraction = SOURCE_FRACTION
    raw_ratio_cap = RAW_RATIO_CAP
    component_center_fraction = COMPONENT_CENTER_FRACTION
    expected_case_count = 17_575
    expected_source_count = 28
    expected_identifier_sha256 = (
        "28f4d1bfd095422f546f0f53134eb4f226963243bc89fcdd78a6d7614bceb509"
    )
    expected_size_bin_counts = (3_946, 2_113, 1_577, 3_515, 3_242, 3_182)

    def _build_loss(self):
        # Increase the false-negative penalty modestly beyond the epoch-160
        # parent's 0.3/0.7 setting. Component-box targets remain unchanged.
        loss = PartialBoxTverskyCELoss(alpha=0.25, beta=0.75)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def _validated_foreground_counts(self, data) -> np.ndarray:
        identifiers = tuple(data.identifiers)
        observed_identifier_sha256 = identifier_inventory_sha256(identifiers)
        if observed_identifier_sha256 != self.expected_identifier_sha256:
            raise ValueError(
                "source-stratified identifier inventory SHA256 mismatch: "
                f"{observed_identifier_sha256}"
            )
        counts = foreground_proxy_counts_from_dataset(data)
        bins = np.asarray([size_bin(int(count)) for count in counts])
        observed_bin_counts = tuple(
            int((bins == bin_index).sum())
            for bin_index in range(len(self.expected_size_bin_counts))
        )
        if observed_bin_counts != self.expected_size_bin_counts:
            raise ValueError(
                "source-stratified foreground proxy bin mismatch: "
                f"{observed_bin_counts}"
            )
        return counts

    def get_dataloaders(self):
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        original_loader = trainer_module.nnUNetDataLoader
        trainer_module.nnUNetDataLoader = component_center_source_balanced_loader_factory(
            original_loader,
            self.source_fraction,
            self.raw_ratio_cap,
            self.component_center_fraction,
            self._validated_foreground_counts,
        )
        try:
            loaders = super().get_dataloaders()
        finally:
            trainer_module.nnUNetDataLoader = original_loader

        training_loader = unwrap_augmenter_loader(loaders[0])
        identifiers = tuple(training_loader._data.identifiers)
        sources = {source_from_identifier(identifier) for identifier in identifiers}
        if len(identifiers) != self.expected_case_count:
            raise ValueError(
                f"expected {self.expected_case_count} training cases, "
                f"found {len(identifiers)}"
            )
        if len(sources) != self.expected_source_count:
            raise ValueError(
                f"expected {self.expected_source_count} sources, found {len(sources)}"
            )
        if getattr(training_loader, "lesion_uniform_center_fraction", None) != 0.25:
            raise ValueError("component-center training loader contract mismatch")
        return loaders

    def on_train_start(self):
        # Avoid attaching this fallback to the active balanced W&B identity.
        STUNetTrainerPanCancer.on_train_start(self)
        if self.local_rank == 0:
            self._initialize_wandb(
                name="stunetb_componentbox_e160_sourcesize_f025_cap3_80ep_20260826",
                run_id="stunetbcompe160sourcesizef025cap320260826",
                partial_label_policy="ignore_outside_union_of_component_bboxes",
                loss_name="trusted_component_box_CE_plus_Tversky_a025_b075",
                extra_config={
                    "parent_epoch": 160,
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "optimizer_restart": True,
                    "optimizer": "SGD_Nesterov_momentum_0.99",
                    "lr_schedule": "polynomial_80_epochs",
                    "case_sampling": "size_preserving_bounded_source_mixture",
                    "loss_change_basis": (
                        "balanced_epoch40_public_proposal_contraction; "
                        "mildly_exceed_epoch160_recall_bias"
                    ),
                    "source_fraction": self.source_fraction,
                    "raw_per_case_ratio_cap": self.raw_ratio_cap,
                    "patch_center_sampling": (
                        "75% foreground-voxel uniform plus 25% annotated-component uniform"
                    ),
                    "component_center_fraction": self.component_center_fraction,
                    "foreground_proxy_bins": [
                        "0-100",
                        "101-500",
                        "501-1000",
                        "1001-3000",
                        "3001-9999",
                        "10000-capped",
                    ],
                    "expected_training_cases": self.expected_case_count,
                    "expected_sources": self.expected_source_count,
                    "expected_identifier_sha256": self.expected_identifier_sha256,
                    "expected_size_bin_counts": self.expected_size_bin_counts,
                },
            )
