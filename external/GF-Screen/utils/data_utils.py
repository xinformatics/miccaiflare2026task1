import math
import os
import fcntl
import pickle
import json
import hashlib
import numpy as np
import torch
import itertools as it
from collections import Counter
from pathlib import Path
from scipy import ndimage
from monai import data, transforms
from monai.data import *
from torch.utils.data import ConcatDataset
from monai.transforms import MapTransform, RandomizableTransform

try:
    import cc3d
except ImportError:  # pragma: no cover - SciPy remains a tested fallback
    cc3d = None


class RemapClippedWindowd(MapTransform):
    """Remap an already scaled clipped window to a narrower target window.

    Persistent GF caches store chest CT values scaled from [-900, 650] to
    [0, 1]. Applying this transform after the random crop reconstructs that
    clipped HU value and maps it to the challenge's [-175, 250] soft-tissue
    window without rebuilding the expensive deterministic cache.
    """

    def __init__(
        self,
        keys,
        source_min=-900.0,
        source_max=650.0,
        target_min=-175.0,
        target_max=250.0,
    ):
        super().__init__(keys)
        if not source_min < source_max or not target_min < target_max:
            raise ValueError("window bounds must be strictly increasing")
        self.source_min = float(source_min)
        self.source_max = float(source_max)
        self.target_min = float(target_min)
        self.target_max = float(target_max)

    def __call__(self, data):
        result = dict(data)
        for key in self.keys:
            value = result[key]
            hu = value * (self.source_max - self.source_min) + self.source_min
            scaled = (hu - self.target_min) / (self.target_max - self.target_min)
            if torch.is_tensor(scaled):
                scaled = scaled.clamp(0.0, 1.0)
            else:
                scaled = np.clip(scaled, 0.0, 1.0)
            result[key] = scaled
        return result


def appearance_augmentation_transforms(args):
    """Return opt-in post-cache CT appearance augmentations."""
    result = []
    if getattr(args, "enable_scale_intensity_aug", False):
        result.append(
            transforms.RandScaleIntensityd(
                keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob
            )
        )
    if getattr(args, "enable_contrast_aug", False):
        gamma_min = float(args.contrast_gamma_min)
        gamma_max = float(args.contrast_gamma_max)
        if not 0.0 < gamma_min <= gamma_max:
            raise ValueError("contrast gamma bounds must satisfy 0 < min <= max")
        result.append(
            transforms.RandAdjustContrastd(
                keys="image",
                prob=args.RandAdjustContrastd_prob,
                gamma=(gamma_min, gamma_max),
            )
        )
    return result


class EmptyCropTrustDataset(torch.utils.data.Dataset):
    """Attach source-level trust without changing PersistentDataset cache keys.

    ``trusted_empty`` is consulted only when a sampled patch contains no
    annotated foreground. Empty patches from the proxy-negative stream are
    fully trusted, while empty patches from partially labelled cancer scans
    require teacher-confirmed background supervision.

    ``complete_label`` is a separate, opt-in declaration.  It is true for the
    healthy proxy stream and for cancer cohorts explicitly listed in a
    versioned completeness file.  Merely containing multiple components is not
    sufficient to set this flag; the caller must name the cohort.

    ``positive_only_label`` preserves pseudo-label provenance through MONAI's
    transforms. Focus uses it to trust pseudo foreground while ignoring
    pseudo-mask background unless the frozen teacher independently marks that
    background safe.
    """

    def __init__(
        self,
        dataset,
        trusted_empty,
        region=None,
        complete_label_cohorts=None,
    ):
        self.dataset = dataset
        self.trusted_empty = bool(trusted_empty)
        self.region = region
        self.complete_label_cohorts = frozenset(complete_label_cohorts or ())
        # Preserve the source-item interface used by cohort balancing.
        self.data = getattr(dataset, "data", None)

    def __len__(self):
        return len(self.dataset)

    def _attach(self, item):
        result = dict(item)
        # ``pseudo_label`` is source-level manifest provenance and is present
        # only on the positive-unlabeled stream. Leaving that asymmetric key in
        # transformed samples makes MONAI's default dict collator fail whenever
        # a batch mixes labeled and pseudo cases. Convert it to the uniform
        # tensor metadata consumed by the loss, then remove the raw key.
        positive_only_label = bool(result.pop("pseudo_label", False))
        result["trusted_empty"] = torch.tensor(self.trusted_empty, dtype=torch.bool)
        complete_label = self.trusted_empty
        if not complete_label and self.region is not None:
            complete_label = (
                _cohort_key(result, self.region) in self.complete_label_cohorts
            )
        result["complete_label"] = torch.tensor(complete_label, dtype=torch.bool)
        result["positive_only_label"] = torch.tensor(
            positive_only_label, dtype=torch.bool
        )
        return result

    def __getitem__(self, index):
        try:
            item = self.dataset[index]
        except Exception as exc:
            # DataLoader normally reports only the transform stack, which made
            # intermittent cache/source failures impossible to attribute.  Add
            # immutable source identity without swallowing or retrying the
            # failure, so a repeated bad case can be excluded precisely.
            source = None
            if self.data is not None and 0 <= int(index) < len(self.data):
                source = self.data[int(index)]
            if isinstance(source, dict):
                identity = (
                    f"name={source.get('name', '<unnamed>')} "
                    f"image={source.get('image', '<missing>')} "
                    f"label={source.get('label', '<missing>')}"
                )
            else:
                identity = f"index={index}"
            raise RuntimeError(f"failed loading dataset item: {identity}") from exc
        if isinstance(item, list):
            return [self._attach(sample) for sample in item]
        if isinstance(item, tuple):
            return tuple(self._attach(sample) for sample in item)
        return self._attach(item)


class ComponentUniformFgBgIndicesd(RandomizableTransform, MapTransform):
    """Build bounded component-positive and body-background crop candidates.

    MONAI's default ``RandCropByPosNegLabeld`` samples uniformly from all
    foreground voxels. Consequently a 10,000-voxel lesion is seen roughly one
    hundred times as often as a 100-voxel lesion in the same scan. At the
    default size power zero, the transform contributes exactly
    ``points_per_component`` flattened indices per connected component, making
    the following random crop uniform over components while retaining random
    centres within each component. A sublinear positive power provides a
    bounded compromise between component-uniform and legacy voxel-uniform
    sampling.

    The transform deliberately implements ``RandomizableTransform`` even
    though its component index itself is deterministic. That keeps it after
    MONAI PersistentDataset's cached deterministic prefix, so the existing
    image/label preprocessing cache remains valid. Small sidecars avoid doing
    connected-components analysis again on later epochs.
    """

    CACHE_VERSION = "component-uniform-v2-cc3d-boxes"
    BACKGROUND_CACHE_VERSION = "component-uniform-v3-body-background"

    def __init__(
        self,
        label_key="label",
        image_key="image",
        name_key="name",
        fg_indices_key="component_fg_indices",
        bg_indices_key="component_bg_indices",
        points_per_component=64,
        size_power=0.0,
        background_points=0,
        cache_dir="",
        legacy_cache_dir="",
    ):
        MapTransform.__init__(self, keys=[label_key, image_key])
        RandomizableTransform.__init__(self, prob=1.0)
        if int(points_per_component) < 1:
            raise ValueError("points_per_component must be at least 1")
        if not 0.0 <= float(size_power) <= 1.0:
            raise ValueError("size_power must be in [0, 1]")
        if int(background_points) < 0:
            raise ValueError("background_points must be non-negative")
        self.label_key = label_key
        self.image_key = image_key
        self.name_key = name_key
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.points_per_component = int(points_per_component)
        self.size_power = float(size_power)
        self.background_points = int(background_points)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.legacy_cache_dir = Path(legacy_cache_dir) if legacy_cache_dir else None

    def _cache_identity(self, name, shape, background):
        cache_version = self.BACKGROUND_CACHE_VERSION if background else self.CACHE_VERSION
        identity = (
            f"{cache_version}|{name}|{tuple(shape)}|"
            f"{self.points_per_component}|{self.size_power:.6f}"
        )
        if background:
            identity += f"|{self.background_points}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".pt"

    def _cache_path(self, name, shape):
        if self.cache_dir is None or not name:
            return None
        return self.cache_dir / self._cache_identity(
            name, shape, self.background_points > 0
        )

    def _legacy_cache_path(self, name, shape):
        if self.legacy_cache_dir is None or not name:
            return None
        return self.legacy_cache_dir / self._cache_identity(name, shape, False)

    def _body_background_indices(self, label, image, name):
        """Sample a fixed background budget without materializing all indices."""
        label_array = torch.as_tensor(label).detach().cpu().numpy()
        image_array = torch.as_tensor(image).detach().cpu().numpy()
        if image_array.ndim != label_array.ndim:
            raise ValueError(
                "image and label must have matching channel/spatial ranks: "
                f"{image_array.shape} versus {label_array.shape}"
            )
        foreground = np.any(label_array > 0, axis=0).reshape(-1)
        flat_image = image_array.reshape(image_array.shape[0], -1)
        if flat_image.shape[1] != foreground.size:
            raise ValueError(
                "image and label spatial shapes must match: "
                f"{image_array.shape[1:]} versus {label_array.shape[1:]}"
            )
        seed_material = (
            f"{self.BACKGROUND_CACHE_VERSION}|background|{name}|"
            f"{tuple(label_array.shape)}|{self.background_points}"
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        rng = np.random.default_rng(seed)
        collected = []
        remaining = self.background_points
        # CT body occupies a substantial fraction of the cached foreground
        # crop. Sampling bounded batches avoids allocating a multi-million-
        # element ``flatnonzero`` result for every training case.
        for _ in range(128):
            if remaining <= 0:
                break
            proposal_count = max(4096, remaining * 4)
            proposed = rng.integers(0, foreground.size, size=proposal_count)
            values = flat_image[:, proposed]
            body = np.any(np.isfinite(values) & (values > 0), axis=0)
            valid = (~foreground[proposed]) & body
            accepted = proposed[valid]
            if accepted.size:
                take = min(remaining, accepted.size)
                collected.append(accepted[:take])
                remaining -= take
        if remaining > 0:
            # Degenerate all-air or nearly full-label volumes are rare. Fall
            # back to annotation-background indices so training remains
            # defined rather than hanging in rejection sampling.
            fallback = np.flatnonzero(~foreground)
            if fallback.size == 0:
                fallback = np.asarray([0], dtype=np.int64)
            collected.append(
                rng.choice(fallback, size=remaining, replace=fallback.size < remaining)
            )
        return np.concatenate(collected).astype(np.int64, copy=False)

    def _compute(self, label, image, name):
        label_array = torch.as_tensor(label).detach().cpu().numpy()
        if label_array.ndim < 2:
            raise ValueError(f"label must have channel and spatial dimensions: {label_array.shape}")
        foreground = np.any(label_array > 0, axis=0)
        seed_material = f"{self.CACHE_VERSION}|{name}|{foreground.shape}"
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            byteorder="little",
            signed=False,
        )
        rng = np.random.default_rng(seed)
        if cc3d is not None and foreground.ndim == 3:
            component_map, component_count = cc3d.connected_components(
                foreground, connectivity=26, return_N=True
            )
            statistics = cc3d.statistics(component_map)
            component_sizes = np.asarray(statistics["voxel_counts"][1:], dtype=np.int64)
            component_boxes = statistics["bounding_boxes"][1:]
        else:
            component_map, component_count = ndimage.label(
                foreground, structure=np.ones((3,) * foreground.ndim, dtype=np.uint8)
            )
            component_sizes = np.bincount(component_map.reshape(-1))[1:].astype(
                np.int64, copy=False
            )
            component_boxes = ndimage.find_objects(component_map)
        if component_count < 1:
            # Cancer streams are expected to contain an annotation. Keep a
            # valid index so MONAI can produce a deterministic empty crop if a
            # malformed case slips through, rather than crashing a worker.
            fg_indices = np.asarray([0], dtype=np.int64)
            component_sizes = np.empty(0, dtype=np.int64)
        else:
            candidates = []
            candidate_counts = component_candidate_counts(
                component_sizes,
                points_per_component=self.points_per_component,
                size_power=self.size_power,
            )
            for component_id, (candidate_count, component_box) in enumerate(
                zip(candidate_counts, component_boxes), start=1
            ):
                if component_box is None:
                    raise RuntimeError(f"missing bounding box for component {component_id}")
                local_coordinates = np.argwhere(
                    component_map[component_box] == component_id
                )
                starts = np.asarray([axis.start for axis in component_box], dtype=np.int64)
                coordinates = local_coordinates + starts
                indices = np.ravel_multi_index(coordinates.T, foreground.shape)
                candidates.append(
                    rng.choice(
                        indices,
                        size=int(candidate_count),
                        replace=indices.size < candidate_count,
                    )
                )
            fg_indices = np.concatenate(candidates).astype(np.int64, copy=False)

        background = ~foreground
        if self.background_points > 0:
            bg_indices = self._body_background_indices(label, image, name)
        else:
            # A single valid index satisfies MONAI's API for positive-only
            # sampling without retaining every background voxel in memory.
            flat_background = np.flatnonzero(background.reshape(-1))
            bg_indices = np.asarray(
                [flat_background[0] if flat_background.size else 0], dtype=np.int64
            )
        return {
            "fg_indices": torch.from_numpy(fg_indices),
            "bg_indices": torch.from_numpy(bg_indices),
            "component_sizes": torch.from_numpy(component_sizes),
        }

    @staticmethod
    def _load_cache_payload(path):
        if path is None or not path.is_file():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError):
            return None

    def _build_payload(self, label, image, name, shape):
        legacy_path = self._legacy_cache_path(name, shape)
        legacy = self._load_cache_payload(legacy_path)
        if (
            self.background_points > 0
            and legacy is not None
            and {"fg_indices", "component_sizes"} <= set(legacy)
        ):
            return {
                "fg_indices": legacy["fg_indices"],
                "component_sizes": legacy["component_sizes"],
                "bg_indices": torch.from_numpy(
                    self._body_background_indices(label, image, name)
                ),
            }
        return self._compute(label, image, name)

    def __call__(self, data):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key]
        name = str(d.get(self.name_key, ""))
        shape = tuple(label.shape)
        cache_path = self._cache_path(name, shape)
        payload = self._load_cache_payload(cache_path)
        if payload is None:
            if cache_path is None:
                payload = self._build_payload(label, image, name, shape)
            else:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Two single-GPU experiment arms often share this sidecar
                # directory.  Atomic replacement prevents corruption but does
                # not prevent both workers from doing the same expensive 3-D
                # connected-component/background computation.  Hold a stable
                # per-case advisory lock, re-check after acquiring it, and let
                # only the first process build the payload.
                lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
                with lock_path.open("a+b") as lock_stream:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                    payload = self._load_cache_payload(cache_path)
                    if payload is None:
                        payload = self._build_payload(label, image, name, shape)
                        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                        torch.save(payload, temporary)
                        os.replace(temporary, cache_path)
        d[self.fg_indices_key] = payload["fg_indices"]
        d[self.bg_indices_key] = payload["bg_indices"]
        return d


def cancer_crop_transforms(args):
    """Return the configured positive/negative cancer crop transforms."""
    fg_key = None
    bg_key = None
    result = []
    if getattr(args, "component_uniform_crop", False):
        fg_key = "component_fg_indices"
        bg_key = "component_bg_indices"
        result.append(
            ComponentUniformFgBgIndicesd(
                points_per_component=args.component_crop_points_per_lesion,
                size_power=args.component_crop_size_power,
                background_points=args.component_background_points,
                cache_dir=args.component_crop_cache_dir,
                legacy_cache_dir=args.component_legacy_cache_dir,
            )
        )
    result.append(
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=args.cancer_positive_weight,
            neg=args.cancer_negative_weight,
            num_samples=args.sw_batch_size,
            image_key="image",
            image_threshold=0,
            fg_indices_key=fg_key,
            bg_indices_key=bg_key,
        )
    )
    return result


def component_candidate_counts(component_sizes, points_per_component=64, size_power=0.0):
    """Allocate a fixed candidate budget with optional sublinear size weighting.

    Power zero is exactly component-uniform. Power one approximates the legacy
    voxel-uniform sampler. Intermediate powers preserve more evidence from
    large lesions while sharply increasing small-lesion exposure.
    """
    sizes = np.asarray(component_sizes, dtype=np.float64)
    if sizes.ndim != 1 or sizes.size < 1 or np.any(sizes <= 0):
        raise ValueError("component_sizes must be a non-empty positive vector")
    if int(points_per_component) < 1:
        raise ValueError("points_per_component must be at least 1")
    if not 0.0 <= float(size_power) <= 1.0:
        raise ValueError("size_power must be in [0, 1]")
    total = int(points_per_component) * sizes.size
    # Reserve one candidate for every component, then distribute the remainder
    # proportionally. This prevents any annotated lesion from disappearing at
    # non-zero powers while keeping the total memory footprint fixed.
    remaining = total - sizes.size
    weights = np.power(sizes, float(size_power))
    raw_extra = weights / weights.sum() * remaining
    counts = np.ones(sizes.size, dtype=np.int64) + np.floor(raw_extra).astype(np.int64)
    leftover = total - int(counts.sum())
    if leftover:
        fractions = raw_extra - np.floor(raw_extra)
        order = np.argsort(-fractions, kind="stable")
        counts[order[:leftover]] += 1
    assert int(counts.sum()) == total
    return counts


def exclude_offline_preprocessing_failures(items, args):
    """Exclude source volumes recorded as unreadable by offline preprocessing."""
    marker = os.path.join(args.cache_dir, "PREPROCESSING_COMPLETE.json")
    if not os.path.isfile(marker):
        return items
    with open(marker) as stream:
        failed = set(json.load(stream).get("failed_names", []))
    if not failed:
        return items
    filtered = [item for item in items if item.get("name") not in failed]
    print(f"Excluded {len(items) - len(filtered)} unreadable offline-preprocessing cases")
    return filtered


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True,
                 num_samples_total=0, weights=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.weights = None
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.double)
            if self.weights.ndim != 1 or len(self.weights) != len(self.dataset):
                raise ValueError("sampling weights must have one value per dataset item")
            if not torch.isfinite(self.weights).all() or torch.any(self.weights <= 0):
                raise ValueError("sampling weights must be finite and strictly positive")
        epoch_size = len(self.dataset) if not num_samples_total else min(len(self.dataset), num_samples_total)
        self.num_samples = int(math.ceil(epoch_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(min(len(self.dataset), self.total_size)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.weights is None:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()[:self.total_size]
            else:
                # Draw one deterministic global stream, then shard it by rank.
                # Replacement is required for genuine weighted sampling and
                # prevents rare cohorts from being exhausted early in an epoch.
                indices = torch.multinomial(
                    self.weights, self.total_size, replacement=True, generator=g
                ).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _dataset_items(dataset):
    """Return MONAI Dataset/PersistentDataset source dictionaries in item order."""
    items = getattr(dataset, "data", None)
    if items is None:
        raise TypeError(f"cannot recover source items from {type(dataset).__name__}")
    return list(items)


def _source_dataset_prefix(name):
    """Recover the source dataset from the challenge case-name convention.

    Broad anatomical families without a hyphen (``WholeBody``, ``Chest``,
    ``Pancreas``...) encode the actual source in the next underscore-delimited
    token.  Dataset names that already contain a hyphen encode the source in
    the first token.  The old first-token-only rule silently collapsed, for
    example, DeepLesion, MSWAL, and autoPETCT into one 6,724-case cohort.
    """
    tokens = str(name or "unknown").split("_")
    first = tokens[0] or "unknown"
    if len(tokens) > 1 and "-" not in first:
        second = tokens[1]
        if second and not second.isdigit():
            return f"{first}_{second}"
    return first


def _cohort_key(item, region):
    """Use the source-dataset prefix as a stable cancer cohort label."""
    return f"{region}:{_source_dataset_prefix(item.get('name', 'unknown'))}"


def load_cohort_priority_weights(path):
    """Load positive source-priority multipliers from a versioned JSON file."""
    if not path:
        return {}
    with open(path) as stream:
        payload = json.load(stream)
    priorities = payload.get("cohort_weights", payload)
    if not isinstance(priorities, dict):
        raise ValueError("cohort priority JSON must contain a cohort_weights object")
    result = {}
    for key, value in priorities.items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid cohort priority for {key}: {value}")
        result[str(key)] = weight
    return result


def load_complete_label_cohorts(path):
    """Load an explicit set of cohorts whose lesion masks are treated as complete."""
    if not path:
        return set()
    with open(path) as stream:
        payload = json.load(stream)
    cohorts = payload.get("complete_label_cohorts", payload)
    if not isinstance(cohorts, list) or not all(
        isinstance(value, str) and value for value in cohorts
    ):
        raise ValueError(
            "complete-label cohort JSON must contain a complete_label_cohorts list"
        )
    if len(cohorts) != len(set(cohorts)):
        raise ValueError("complete-label cohort JSON contains duplicate cohorts")
    return set(cohorts)


def load_positive_pseudo_datalist(path, base_dir, repeat=1):
    """Load a positive-only pseudo manifest without granting empty-mask trust."""
    if not path:
        return []
    if int(repeat) < 1:
        raise ValueError("pseudo-label exposure repeat must be at least 1")
    items = load_decathlon_datalist(path, True, "training", base_dir=base_dir)
    if any(not item.get("pseudo_label", False) for item in items):
        raise ValueError(
            "every pseudo-label training entry must declare pseudo_label=true"
        )
    if any(not item.get("image") or not item.get("label") for item in items):
        raise ValueError("pseudo-label entries require both image and label paths")
    # These immutable-manifest identity fields are validated before this
    # loader is called, but must not enter MONAI's transform dictionary.
    # Spacingd treats every ``image_*``/``label_*`` key as a metadata mapping;
    # scalar digest/size values therefore collide with that reserved namespace.
    for item in items:
        item.pop("image_sha256", None)
        item.pop("image_bytes", None)
        item.pop("label_sha256", None)
        item.pop("label_bytes", None)
    return items * int(repeat)


def build_cohort_sampling_weights(
    dataset_groups, power=0.0, max_weight=5.0, priority_weights=None
):
    """Build capped inverse-frequency weights without altering the default path.

    ``dataset_groups`` is an ordered sequence of ``(region, dataset)`` pairs.
    The synthetic healthy-proxy stream is deliberately assigned unit weight;
    only annotated lesion cohorts are rebalanced.
    """
    if power < 0:
        raise ValueError("cohort balance power must be non-negative")
    if max_weight < 1:
        raise ValueError("cohort balance max weight must be at least 1")

    keys = []
    for region, dataset in dataset_groups:
        items = _dataset_items(dataset)
        if region == "healthy_proxy":
            keys.extend([region] * len(items))
        else:
            keys.extend(_cohort_key(item, region) for item in items)

    counts = Counter(key for key in keys if key != "healthy_proxy")
    if not counts:
        return torch.ones(len(keys), dtype=torch.double), Counter(keys), {}
    largest = max(counts.values())
    priority_weights = priority_weights or {}
    unknown = sorted(set(priority_weights) - set(counts))
    if unknown:
        raise ValueError(f"cohort priorities do not match the dataset: {unknown}")
    raw_weights = {
        key: min(float(max_weight), (largest / count) ** float(power))
        * float(priority_weights.get(key, 1.0))
        for key, count in counts.items()
    }
    # Preserve the aggregate lesion-vs-proxy sampling ratio. Rebalancing must
    # change only which annotated cohort supplies a lesion case, not silently
    # dilute the negative/proxy stream by increasing total lesion mass.
    lesion_count = sum(counts.values())
    lesion_mass = sum(count * raw_weights[key] for key, count in counts.items())
    normalization = lesion_count / lesion_mass
    cohort_weights = {
        key: raw_weight * normalization for key, raw_weight in raw_weights.items()
    }
    cohort_weights["healthy_proxy"] = 1.0
    weights = torch.tensor([cohort_weights[key] for key in keys], dtype=torch.double)
    return weights, Counter(keys), cohort_weights


def _print_cohort_sampling_summary(counts, cohort_weights):
    total_mass = sum(count * cohort_weights[key] for key, count in counts.items())
    print("Cohort-balanced case sampling enabled")
    for key in sorted(counts, key=lambda value: (-counts[value], value)):
        expected = counts[key] * cohort_weights[key] / total_mass if total_mass else 0.0
        print(
            f"  {key}: n={counts[key]} weight={cohort_weights[key]:.3f} "
            f"expected={100.0 * expected:.2f}%"
        )


def get_abdomen_ds(args):
    data_dir = args.data_dir
    datalist_json = args.abdomen_json

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            # Negative cancer patches are opt-in. Their nominal background is
            # not trusted directly; the loss uses only frozen-teacher
            # confident background so unannotated disease is not erased.
            *cancer_crop_transforms(args),
            *appearance_augmentation_transforms(args),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_list = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    pseudo_json = str(getattr(args, "pseudo_abdomen_json", "") or "")
    if pseudo_json:
        repeat = int(getattr(args, "pseudo_abdomen_repeat", 1))
        pseudo_list = load_positive_pseudo_datalist(
            pseudo_json, base_dir=data_dir, repeat=1
        )
        datalist.extend(pseudo_list * repeat)
        print(
            "Added positive-only abdominal pseudo labels:",
            len(pseudo_list),
            "cases x",
            repeat,
            "exposure",
        )
    datalist = exclude_offline_preprocessing_failures(datalist, args)
    val_list = exclude_offline_preprocessing_failures(val_list, args)

    if args.use_persistent_dataset:
        print('use persistent')
        train_ds = PersistentDataset(data=datalist,
                                     transform=train_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     weights_only=False,
                                     cache_dir=os.path.join(args.cache_dir, "abdomen"))

        val_ds = PersistentDataset(data=val_list,
                                   transform=val_transform,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   weights_only=False,
                                   cache_dir=os.path.join(args.cache_dir, "abdomen"))
    else:
        train_ds = data.Dataset(
            data=datalist, transform=train_transform)
        val_ds = data.Dataset(data=val_list, transform=val_transform)
    return train_ds, val_ds


def get_chest_ds(args):
    data_dir = args.data_dir
    datalist_json = args.chest_json

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-900.0, a_max=650.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),


            *cancer_crop_transforms(args),
            *(
                [RemapClippedWindowd(keys="image")]
                if getattr(args, "harmonize_cached_chest_window", False)
                else []
            ),
            *appearance_augmentation_transforms(args),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-900.0, a_max=650.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_list = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    datalist = exclude_offline_preprocessing_failures(datalist, args)
    val_list = exclude_offline_preprocessing_failures(val_list, args)

    if args.use_persistent_dataset:
        print('use persistent')
        train_ds = PersistentDataset(data=datalist,
                                     transform=train_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     weights_only=False,
                                     cache_dir=os.path.join(args.cache_dir, "chest"))

        val_ds = PersistentDataset(data=val_list,
                                   transform=val_transform,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   weights_only=False,
                                   cache_dir=os.path.join(args.cache_dir, "chest"))
    else:
        train_ds = data.Dataset(
            data=datalist, transform=train_transform)
        val_ds = data.Dataset(data=val_list, transform=val_transform)
    return train_ds, val_ds


def get_nolesion_ds(args):
    data_dir = args.data_dir
    datalist_json = args.healthy_json

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            transforms.RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(args.roi_x, args.roi_y, args.roi_z),
                num_samples=args.sw_batch_size,
                random_size=False
            ),
            *appearance_augmentation_transforms(args),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),

            Filter_nolesion_Labels(keys='label')
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),

            Filter_nolesion_Labels(keys='label')
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_list = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    datalist = exclude_offline_preprocessing_failures(datalist, args)
    val_list = exclude_offline_preprocessing_failures(val_list, args)

    if args.use_persistent_dataset:
        print('use persistent')
        train_ds = PersistentDataset(data=datalist * args.healthy_repeat,
                                     transform=train_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     weights_only=False,
                                     cache_dir=os.path.join(args.cache_dir, "healthy"))

        val_ds = PersistentDataset(data=val_list,
                                   transform=val_transform,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   weights_only=False,
                                   cache_dir=os.path.join(args.cache_dir, "healthy"))
    else:
        train_ds = data.Dataset(
            data=datalist, transform=train_transform)
        val_ds = data.Dataset(data=val_list, transform=val_transform)
    return train_ds, val_ds


class Filter_nolesion_Labels(MapTransform):
    """Filter unsed label.
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            new_img = d[key].clone()
            new_img[d[key] >= 0] = 0
            new_img[d[key] < 0] = 0
            d[key] = new_img.float()

        return d


def get_loader(args):
    abdomen_train_ds, abdomen_val_ds = get_abdomen_ds(args)
    chest_train_ds, chest_val_ds = get_chest_ds(args)
    nolesion_train_ds, nolesion_val_ds = get_nolesion_ds(args)

    complete_label_cohorts = load_complete_label_cohorts(
        getattr(args, "complete_label_cohorts_json", "")
    )
    known_cohorts = {
        *(_cohort_key(item, "abdomen") for item in _dataset_items(abdomen_train_ds)),
        *(_cohort_key(item, "chest") for item in _dataset_items(chest_train_ds)),
    }
    unknown_complete_cohorts = sorted(complete_label_cohorts - known_cohorts)
    if unknown_complete_cohorts:
        raise ValueError(
            "complete-label cohorts do not match the dataset: "
            f"{unknown_complete_cohorts}"
        )

    # Wrap after PersistentDataset construction so trust metadata does not
    # invalidate or duplicate the expensive deterministic preprocessing cache.
    abdomen_train_ds = EmptyCropTrustDataset(
        abdomen_train_ds,
        trusted_empty=False,
        region="abdomen",
        complete_label_cohorts=complete_label_cohorts,
    )
    chest_train_ds = EmptyCropTrustDataset(
        chest_train_ds,
        trusted_empty=False,
        region="chest",
        complete_label_cohorts=complete_label_cohorts,
    )
    nolesion_train_ds = EmptyCropTrustDataset(
        nolesion_train_ds,
        trusted_empty=True,
        region="healthy_proxy",
        complete_label_cohorts=complete_label_cohorts,
    )

    train_ds = ConcatDataset([abdomen_train_ds, chest_train_ds, nolesion_train_ds])
    val_ds = ConcatDataset([abdomen_val_ds, chest_val_ds, nolesion_val_ds])

    sampling_weights = None
    cohort_priorities = load_cohort_priority_weights(
        getattr(args, "cohort_priority_json", "")
    )
    if args.cohort_balance_power > 0 or cohort_priorities:
        sampling_weights, cohort_counts, cohort_weights = build_cohort_sampling_weights(
            [
                ("abdomen", abdomen_train_ds),
                ("chest", chest_train_ds),
                ("healthy_proxy", nolesion_train_ds),
            ],
            power=args.cohort_balance_power,
            max_weight=args.cohort_balance_max_weight,
            priority_weights=cohort_priorities,
        )
        if len(sampling_weights) != len(train_ds):
            raise RuntimeError("cohort sampling weights do not match concatenated dataset")
        if getattr(args, "rank", 0) == 0:
            _print_cohort_sampling_summary(cohort_counts, cohort_weights)

    train_sampler = None
    if args.distributed:
        train_sampler = Sampler(
            train_ds,
            num_samples_total=args.samples_per_epoch,
            weights=sampling_weights,
        )
    elif args.samples_per_epoch or sampling_weights is not None:
        # Use the same explicit epoch-seeded sampler on one GPU.  RandomSampler
        # draws its iterator seed from the process-global RNG, whose state also
        # depends on model initialization and augmentations; that makes the
        # actual case stream impossible to reconstruct from run metadata.
        train_sampler = Sampler(
            train_ds,
            num_replicas=1,
            rank=0,
            num_samples_total=args.samples_per_epoch,
            weights=sampling_weights,
        )
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
    )
    loader = [train_loader, val_loader]

    return loader


def get_val_loader(args):
    abdomen_train_ds, abdomen_val_ds = get_abdomen_ds(args)
    chest_train_ds, chest_val_ds = get_chest_ds(args)

    val_ds = ConcatDataset([abdomen_val_ds, chest_val_ds])

    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=True, num_workers=args.workers, sampler=None, pin_memory=True
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-900.0, a_max=650.0, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                                   mode="constant"),
        ]
    )

    return val_loader, val_transform
