# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
import torch.nn.functional as F
from monai.data import decollate_batch
from scipy import ndimage
from models.GF_Screen_RL import teacher_safe_glance_masks
from fault_tolerant_wandb import (
    safe_wandb_define_metrics,
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
    stable_wandb_run_id,
)


CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)
ANNOTATED_RECALL_BINS_MM3 = (
    ("lt100", 0.0, 100.0),
    ("100to500", 100.0, 500.0),
    ("500to1000", 500.0, 1000.0),
    ("1000to3000", 1000.0, 3000.0),
    ("ge3000", 3000.0, float("inf")),
)


def atomic_torch_save(payload, destination):
    """Publish a torch artifact only after serialization completes."""

    destination = os.fspath(destination)
    directory = os.path.dirname(destination) or "."
    basename = os.path.basename(destination)
    temporary = os.path.join(directory, f".{basename}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_copyfile(source, destination):
    """Atomically publish a copy without exposing a partial destination."""

    destination = os.fspath(destination)
    directory = os.path.dirname(destination) or "."
    basename = os.path.basename(destination)
    temporary = os.path.join(directory, f".{basename}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def annotated_recall_counts(
    prediction,
    reference,
    voxel_mm3,
    minimum_component_overlap_fraction=0.10,
):
    """Count only label-safe lesion-recall events on a partial reference.

    Challenge training masks may omit secondary lesions, so prediction-only
    voxels cannot be called false positives. Annotated foreground remains valid
    positive evidence: voxel recall and whether each known connected component
    receives at least 10% overlap are therefore safe diagnostics.
    """
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if prediction.ndim != 3 or reference.ndim != 3:
        raise ValueError("annotated recall expects two 3D masks")
    if prediction.shape != reference.shape:
        raise ValueError("prediction/reference shape mismatch")
    voxel_mm3 = float(voxel_mm3)
    if not np.isfinite(voxel_mm3) or voxel_mm3 <= 0:
        raise ValueError("voxel_mm3 must be finite and positive")
    overlap_threshold = float(minimum_component_overlap_fraction)
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("component overlap threshold must lie in (0, 1]")

    counts = {
        "annotated_voxel_tp": int(np.count_nonzero(prediction & reference)),
        "annotated_voxel_total": int(np.count_nonzero(reference)),
        "annotated_components_detected": 0,
        "annotated_components_total": 0,
    }
    for tag, _lower, _upper in ANNOTATED_RECALL_BINS_MM3:
        counts[f"annotated_components_{tag}_detected"] = 0
        counts[f"annotated_components_{tag}_total"] = 0

    component_map, component_count = ndimage.label(reference, CONNECTIVITY_26)
    for component_id in range(1, component_count + 1):
        component = component_map == component_id
        component_voxels = int(np.count_nonzero(component))
        if component_voxels == 0:
            continue
        overlap_fraction = (
            float(np.count_nonzero(prediction & component)) / component_voxels
        )
        detected = overlap_fraction >= overlap_threshold
        volume_mm3 = component_voxels * voxel_mm3
        counts["annotated_components_total"] += 1
        counts["annotated_components_detected"] += int(detected)
        for tag, lower, upper in ANNOTATED_RECALL_BINS_MM3:
            if lower <= volume_mm3 < upper:
                counts[f"annotated_components_{tag}_total"] += 1
                counts[f"annotated_components_{tag}_detected"] += int(detected)
                break
    return counts


def component_box_targets(crop_labels):
    """Build one component-localization target for each positive Glance crop.

    Component-uniform sampling chooses foreground points independently from
    every annotated lesion. If another lesion also falls inside the crop, the
    component whose bounding-box center is closest to the crop center is the
    sampled proposal target; this avoids an enclosing box spanning two remote
    lesions. Coordinates are normalized inclusive bounds in D/H/W order.
    """
    if crop_labels.ndim != 5 or crop_labels.shape[1] != 1:
        raise ValueError("crop_labels must have shape [N,1,D,H,W]")
    spatial_shape = np.asarray(crop_labels.shape[2:], dtype=np.float64)
    denominators = np.maximum(spatial_shape - 1.0, 1.0)
    crop_center = (spatial_shape - 1.0) / 2.0
    targets = np.zeros((crop_labels.shape[0], 6), dtype=np.float32)
    valid = np.zeros(crop_labels.shape[0], dtype=bool)
    masks = crop_labels.detach().gt(0.5).cpu().numpy()[:, 0]
    for batch_index, mask in enumerate(masks):
        component_map, component_count = ndimage.label(mask, CONNECTIVITY_26)
        if component_count < 1:
            continue
        candidates = []
        for component_slice in ndimage.find_objects(component_map):
            if component_slice is None:
                continue
            starts = np.asarray(
                [axis.start for axis in component_slice], dtype=np.float64
            )
            ends = np.asarray(
                [axis.stop - 1 for axis in component_slice], dtype=np.float64
            )
            center = (starts + ends) / 2.0
            candidates.append((float(np.square(center - crop_center).sum()), starts, ends))
        if not candidates:
            continue
        _, starts, ends = min(candidates, key=lambda value: value[0])
        targets[batch_index, :3] = starts / denominators
        targets[batch_index, 3:] = ends / denominators
        valid[batch_index] = True
    return (
        torch.as_tensor(targets, device=crop_labels.device, dtype=torch.float32),
        torch.as_tensor(valid, device=crop_labels.device, dtype=torch.bool),
    )


def glance_component_box_loss(box_logits, crop_labels):
    """Smooth-L1 auxiliary localization loss on annotation-positive crops."""
    if box_logits.ndim != 2 or box_logits.shape[1] != 6:
        raise ValueError("box_logits must have shape [N,6]")
    if box_logits.shape[0] != crop_labels.shape[0]:
        raise ValueError("box logits and crop labels must have matching batches")
    targets, valid = component_box_targets(crop_labels)
    predictions = box_logits.sigmoid()
    zero = box_logits.sum() * 0.0
    if not valid.any():
        return zero, {
            "glance_box_loss": zero.detach(),
            "glance_box_positive_crops": valid.sum().detach(),
            "glance_box_iou": zero.detach(),
            "glance_box_center_error": zero.detach(),
        }
    loss = F.smooth_l1_loss(predictions[valid], targets[valid])
    predicted_start = torch.minimum(predictions[valid, :3], predictions[valid, 3:])
    predicted_end = torch.maximum(predictions[valid, :3], predictions[valid, 3:])
    target_start = targets[valid, :3]
    target_end = targets[valid, 3:]
    intersection = (
        (torch.minimum(predicted_end, target_end) - torch.maximum(predicted_start, target_start))
        .clamp_min(0.0)
        .prod(1)
    )
    predicted_volume = (predicted_end - predicted_start).clamp_min(0.0).prod(1)
    target_volume = (target_end - target_start).clamp_min(0.0).prod(1)
    union = (predicted_volume + target_volume - intersection).clamp_min(1e-6)
    center_error = torch.linalg.vector_norm(
        (predicted_start + predicted_end) / 2.0
        - (target_start + target_end) / 2.0,
        dim=1,
    )
    return loss, {
        "glance_box_loss": loss.detach(),
        "glance_box_positive_crops": valid.sum().detach(),
        "glance_box_iou": (intersection / union).mean().detach(),
        "glance_box_center_error": center_error.mean().detach(),
    }


def component_center_targets(crop_labels, output_shape):
    """Rasterize one dense centre target for every annotated component.

    Targets live on the quarter-resolution Glance feature map.  Every
    connected component receives at least one positive cell, irrespective of
    its voxel volume, which avoids erasing tiny lesions during label
    downsampling.
    """
    if crop_labels.ndim != 5 or crop_labels.shape[1] != 1:
        raise ValueError("crop_labels must have shape [N,1,D,H,W]")
    output_shape = tuple(int(value) for value in output_shape)
    if len(output_shape) != 3 or any(value < 1 for value in output_shape):
        raise ValueError("output_shape must contain three positive dimensions")
    input_shape = np.asarray(crop_labels.shape[2:], dtype=np.float64)
    output_extent = np.maximum(np.asarray(output_shape) - 1, 0)
    input_extent = np.maximum(input_shape - 1.0, 1.0)
    targets = np.zeros(
        (crop_labels.shape[0], 1, *output_shape), dtype=np.float32
    )
    positive_crops = np.zeros(crop_labels.shape[0], dtype=bool)
    masks = crop_labels.detach().gt(0.5).cpu().numpy()[:, 0]
    for batch_index, mask in enumerate(masks):
        component_map, component_count = ndimage.label(mask, CONNECTIVITY_26)
        if component_count < 1:
            continue
        positive_crops[batch_index] = True
        for component_slice in ndimage.find_objects(component_map):
            if component_slice is None:
                continue
            starts = np.asarray(
                [axis.start for axis in component_slice], dtype=np.float64
            )
            ends = np.asarray(
                [axis.stop - 1 for axis in component_slice], dtype=np.float64
            )
            center = (starts + ends) / 2.0
            output_center = np.rint(center / input_extent * output_extent).astype(
                np.int64
            )
            output_center = np.minimum(
                np.maximum(output_center, 0), np.asarray(output_shape) - 1
            )
            targets[(batch_index, 0, *output_center.tolist())] = 1.0
    return (
        torch.as_tensor(targets, device=crop_labels.device),
        torch.as_tensor(
            positive_crops, device=crop_labels.device, dtype=torch.bool
        ),
    )


def glance_component_center_loss(
    center_logits,
    crop_labels,
    complete_label=None,
    focal_gamma=2.0,
):
    """Partial-label-safe dense centerness objective for Glance.

    Annotated lesion centres are always positive.  Background cells are
    negatives only for documentation-verified complete masks (including the
    healthy stream); every other cell in a partially labelled crop is ignored
    so an unsegmented lesion cannot become a false-negative target.
    """
    if center_logits.ndim != 5 or center_logits.shape[1] != 1:
        raise ValueError("center_logits must have shape [N,1,D,H,W]")
    if center_logits.shape[0] != crop_labels.shape[0]:
        raise ValueError("center logits and crop labels must have matching batches")
    if float(focal_gamma) < 0:
        raise ValueError("focal_gamma must be non-negative")
    targets, positive_crops = component_center_targets(
        crop_labels, center_logits.shape[2:]
    )
    positive = targets.bool()
    complete = torch.zeros(
        center_logits.shape[0], device=center_logits.device, dtype=torch.bool
    )
    if complete_label is not None:
        complete = torch.as_tensor(
            complete_label, device=center_logits.device, dtype=torch.bool
        ).flatten()
        if complete.numel() != center_logits.shape[0]:
            raise ValueError(
                "complete_label must have one value per Glance crop "
                f"({complete.numel()} != {center_logits.shape[0]})"
            )
    supervised_complete = complete.reshape(-1, 1, 1, 1, 1).expand_as(positive)
    negative = supervised_complete & (~positive)
    probability = center_logits.sigmoid()
    eps = 1e-6
    zero = center_logits.sum() * 0.0
    positive_loss = (
        -(
            (1.0 - probability[positive]).pow(float(focal_gamma))
            * torch.log(probability[positive].clamp_min(eps))
        ).mean()
        if positive.any()
        else zero
    )
    negative_loss = (
        -(
            probability[negative].pow(float(focal_gamma))
            * torch.log((1.0 - probability[negative]).clamp_min(eps))
        ).mean()
        if negative.any()
        else zero
    )
    peak_errors = []
    output_shape = center_logits.shape[2:]
    denominator = torch.as_tensor(
        [max(value - 1, 1) for value in output_shape],
        device=center_logits.device,
        dtype=torch.float32,
    )
    for batch_index in positive_crops.nonzero(as_tuple=False).flatten():
        flat_peak = center_logits[batch_index, 0].reshape(-1).argmax()
        plane = output_shape[1] * output_shape[2]
        peak = torch.stack(
            (
                flat_peak // plane,
                (flat_peak % plane) // output_shape[2],
                flat_peak % output_shape[2],
            )
        ).float()
        target_centers = positive[batch_index, 0].nonzero(as_tuple=False).float()
        peak_errors.append(
            torch.linalg.vector_norm(
                (target_centers - peak.unsqueeze(0)) / denominator, dim=1
            ).min()
        )
    peak_error = torch.stack(peak_errors).mean() if peak_errors else zero
    loss = positive_loss + negative_loss
    return loss, {
        "glance_center_loss": loss.detach(),
        "glance_center_positive_cells": positive.sum().detach(),
        "glance_center_negative_cells": negative.sum().detach(),
        "glance_center_positive_crops": positive_crops.sum().detach(),
        "glance_center_peak_error": peak_error.detach(),
    }


def teacher_safe_glance_loss(
    actions,
    annotated_labels,
    teacher_logits,
    pseudo_positive_threshold=0.95,
    pseudo_negative_threshold=0.05,
    pseudo_positive_min_voxels=64,
    complete_label=None,
):
    """Partial-label-safe crop classification for the Glance gate.

    This objective permits hard negative crops to be sampled from partially
    labelled cancer scans without assuming that every crop missing the one
    annotated primary lesion is truly background. Keep annotated crops
    positive, promote only sizeable high-confidence frozen-Focus islands,
    retain only teacher-confident empty crops as negatives, and ignore the
    uncertain remainder.
    """
    if not 0.0 <= pseudo_negative_threshold < pseudo_positive_threshold <= 1.0:
        raise ValueError("Glance teacher thresholds must satisfy 0 <= negative < positive <= 1")
    if pseudo_positive_min_voxels < 1:
        raise ValueError("Glance pseudo-positive minimum voxels must be at least 1")
    if actions.shape[0] != annotated_labels.numel() or actions.shape[0] != teacher_logits.shape[0]:
        raise ValueError("Glance actions, labels, and teacher logits must have matching batches")

    probability = actions.softmax(1)[:, 1]
    (
        annotated_positive,
        pseudo_positive,
        safe_negative,
        valid,
        teacher_positive_voxels,
    ) = teacher_safe_glance_masks(
        annotated_labels,
        teacher_logits,
        pseudo_positive_threshold=pseudo_positive_threshold,
        pseudo_negative_threshold=pseudo_negative_threshold,
        pseudo_positive_min_voxels=pseudo_positive_min_voxels,
    )
    complete_safe_negative = torch.zeros_like(safe_negative)
    if complete_label is not None:
        complete = torch.as_tensor(
            complete_label, device=actions.device, dtype=torch.bool
        ).flatten()
        if complete.numel() != actions.shape[0]:
            raise ValueError(
                "complete_label must have one value per Glance crop "
                f"({complete.numel()} != {actions.shape[0]})"
            )
        # A complete mask is authoritative: an annotation-negative crop is a
        # real negative even when the frozen teacher is uncertain or wrong.
        # Conversely, teacher pseudo positives cannot override a complete mask.
        complete_safe_negative = complete & (~annotated_positive)
        pseudo_positive &= ~complete
        safe_negative = (safe_negative & (~complete)) | complete_safe_negative
        valid = annotated_positive | pseudo_positive | safe_negative
    positive = annotated_positive | pseudo_positive

    eps = 1e-6
    zero = probability.sum() * 0.0
    positive_loss = (
        -torch.log(probability[positive].clamp_min(eps)).mean()
        if positive.any() else zero
    )
    negative_loss = (
        -torch.log((1.0 - probability[safe_negative]).clamp_min(eps)).mean()
        if safe_negative.any() else zero
    )
    loss = positive_loss + negative_loss
    return loss, {
        "glance_annotated_positive_crops": annotated_positive.sum().detach(),
        "glance_pseudo_positive_crops": pseudo_positive.sum().detach(),
        "glance_safe_negative_crops": safe_negative.sum().detach(),
        "glance_complete_safe_negative_crops": complete_safe_negative.sum().detach(),
        "glance_ignored_crops": (~valid).sum().detach(),
        "glance_pseudo_positive_voxels": teacher_positive_voxels[
            pseudo_positive
        ].sum().detach(),
    }


def glance_ignored_distillation_loss(
    actions,
    reference_actions,
    annotated_labels,
    teacher_logits,
    pseudo_positive_threshold=0.95,
    pseudo_negative_threshold=0.05,
    pseudo_positive_min_voxels=64,
    complete_label=None,
):
    """Preserve the source Glance policy only on uncertain partial-label crops.

    Confirmed annotation/teacher positives and trusted negatives remain free to
    learn from the direct classification objective. Crops for which neither
    source is trustworthy receive a KL target from the frozen starting Glance
    model instead of an invented background label.
    """
    if actions.shape != reference_actions.shape:
        raise ValueError("Glance and reference actions must have matching shapes")
    annotated_positive, pseudo_positive, safe_negative, valid, _ = (
        teacher_safe_glance_masks(
            annotated_labels,
            teacher_logits,
            pseudo_positive_threshold=pseudo_positive_threshold,
            pseudo_negative_threshold=pseudo_negative_threshold,
            pseudo_positive_min_voxels=pseudo_positive_min_voxels,
        )
    )
    if complete_label is not None:
        complete = torch.as_tensor(
            complete_label, device=actions.device, dtype=torch.bool
        ).flatten()
        if complete.numel() != actions.shape[0]:
            raise ValueError(
                "complete_label must have one value per Glance crop "
                f"({complete.numel()} != {actions.shape[0]})"
            )
        complete_safe_negative = complete & (~annotated_positive)
        pseudo_positive &= ~complete
        safe_negative = (safe_negative & (~complete)) | complete_safe_negative
        valid = annotated_positive | pseudo_positive | safe_negative
    ignored = ~valid
    zero = actions.sum() * 0.0
    loss = (
        F.kl_div(
            F.log_softmax(actions[ignored], dim=1),
            F.softmax(reference_actions[ignored].detach(), dim=1),
            reduction="batchmean",
        )
        if ignored.any()
        else zero
    )
    return loss, {
        "glance_distillation_loss": loss.detach(),
        "glance_distilled_crops": ignored.sum().detach(),
    }


def component_box_trusted_mask(ground_truth, margins):
    """Return the union of independently expanded 3D component boxes.

    Connected components are found on CPU because the 96x96x64 training masks
    are small and SciPy's implementation is deterministic. Only a single
    compact binary batch is transferred from GPU; the returned mask is created
    directly on the original device.
    """
    trusted = torch.zeros_like(ground_truth, dtype=torch.bool)
    component_total = 0
    masks = ground_truth.detach().gt(0.5).cpu().numpy()
    for batch_index, mask in enumerate(masks):
        component_map, component_count = ndimage.label(mask, CONNECTIVITY_26)
        if component_count == 0:
            trusted[batch_index] = True
            continue
        component_total += component_count
        for component_slices in ndimage.find_objects(component_map):
            if component_slices is None:
                continue
            expanded = []
            for axis, (component_slice, margin) in enumerate(zip(component_slices, margins)):
                start = max(0, int(component_slice.start) - margin)
                stop = min(ground_truth.shape[axis + 1], int(component_slice.stop) + margin)
                expanded.append(slice(start, stop))
            trusted[(batch_index, *expanded)] = True
    mean_components = ground_truth.new_tensor(
        component_total / max(1, ground_truth.shape[0])
    )
    return trusted, mean_components


def partial_label_teacher_loss(student_logits, labels, teacher_logits, args):
    """Positive-unlabeled adaptation anchored to the released model.

    Primary-lesion voxels are trusted. On lesion-bearing crops, only very
    confident teacher predictions outside the annotation become weak positive
    targets; uncertain voxels are ignored instead of being mislabeled as
    background. Truly empty training crops remain strong negatives. A 2 mm
    in-plane boundary band aligns the optimization signal with challenge NSD.
    """
    eps = 1e-6
    probability = student_logits.softmax(1)[:, 1]
    teacher_probability = teacher_logits.softmax(1)[:, 1].detach()
    ground_truth = labels[:, 0].float()
    lesion_crop = ground_truth.flatten(1).sum(1).gt(0).view(-1, 1, 1, 1)
    unlabeled = ground_truth.lt(0.5)

    pseudo_positive = (
        unlabeled & lesion_crop & teacher_probability.ge(args.pseudo_positive_threshold)
    )
    confident_negative = (
        unlabeled & teacher_probability.le(args.pseudo_negative_threshold)
    )
    known_healthy = unlabeled & (~lesion_crop)
    merged_target = torch.maximum(ground_truth, pseudo_positive.float())

    # Alpha weights false positives and beta weights false negatives. Keeping
    # them configurable supports leaderboard-guided precision/recall tuning.
    reduce_dims = (1, 2, 3)
    tp = (probability * merged_target).sum(reduce_dims)
    fp = (probability * (1.0 - merged_target)).sum(reduce_dims)
    fn = ((1.0 - probability) * merged_target).sum(reduce_dims)
    tversky = 1.0 - ((tp + eps) / (
        tp + args.tversky_alpha * fp + args.tversky_beta * fn + eps
    )).mean()

    voxel_bce = F.binary_cross_entropy(probability.clamp(eps, 1.0 - eps), merged_target, reduction="none")
    supervision_weight = (
        ground_truth
        + pseudo_positive.float() * args.pseudo_positive_weight
        + confident_negative.float() * args.confident_negative_weight
        + known_healthy.float()
    ).clamp_max(1.0)
    weighted_bce = (voxel_bce * supervision_weight).sum() / supervision_weight.sum().clamp_min(1.0)

    # Spacing is 1x1x3 mm. A five-voxel in-plane morphological band is a
    # practical differentiable proxy for the challenge's 2 mm surface band.
    gt_5d = ground_truth.unsqueeze(1)
    dilated = F.max_pool3d(gt_5d, kernel_size=(5, 5, 1), stride=1, padding=(2, 2, 0))
    eroded = -F.max_pool3d(-gt_5d, kernel_size=(5, 5, 1), stride=1, padding=(2, 2, 0))
    boundary_band = (dilated - eroded).gt(0).squeeze(1).float()
    boundary = (voxel_bce * boundary_band).sum() / boundary_band.sum().clamp_min(1.0)

    teacher_confident = lesion_crop & unlabeled & (
        teacher_probability.ge(args.pseudo_positive_threshold)
        | teacher_probability.le(args.pseudo_negative_threshold)
    )
    consistency = (
        (probability - teacher_probability).square() * teacher_confident.float()
    ).sum() / teacher_confident.float().sum().clamp_min(1.0)

    total = (
        0.55 * tversky
        + 0.30 * weighted_bce
        + args.boundary_weight * boundary
        + args.teacher_consistency_weight * consistency
    )
    return total, {
        "tversky": tversky.detach(),
        "weighted_bce": weighted_bce.detach(),
        "boundary": boundary.detach(),
        "consistency": consistency.detach(),
        "pseudo_positive_voxels": pseudo_positive.sum().detach(),
    }


def bbox_partial_label_loss(
    student_logits,
    labels,
    args,
    teacher_logits=None,
    trusted_empty=None,
    positive_only=None,
):
    """Target-box loss for scans whose non-target lesions are unannotated.

    The network still sees the complete 96x96x64 CT context. For a crop with an
    annotated lesion, however, only an automatically expanded 3D box around
    that target contributes to the loss. This prevents distant, unannotated
    lesions from being learned as background. Empty crops from the trusted
    healthy proxy set remain fully supervised negatives. Empty crops sampled
    from partially labelled cancer scans use only frozen-teacher confident
    background voxels; uncertain voxels and teacher foreground are ignored.

    When a frozen teacher is supplied, high-confidence teacher foreground
    outside the provided target is removed from the trusted-negative mask.
    Optionally, a separately weighted positive-only BCE term can promote those
    voxels without allowing pseudo-labels into the Tversky denominator or the
    trusted-background loss.  Its default weight is zero, preserving the
    original bounding-box objective exactly.
    """
    eps = 1e-6
    probability = student_logits.softmax(1)[:, 1]
    ground_truth = labels[:, 0].float()
    trusted = torch.zeros_like(ground_truth, dtype=torch.bool)
    lesion_crop = ground_truth.flatten(1).sum(1).gt(0)
    if trusted_empty is None:
        # Exact backward-compatible behavior for callers/checkpoints created
        # before source trust metadata was introduced.
        trusted_empty = torch.ones_like(lesion_crop, dtype=torch.bool)
    else:
        trusted_empty = torch.as_tensor(
            trusted_empty, device=ground_truth.device, dtype=torch.bool
        ).flatten()
        if trusted_empty.numel() != ground_truth.shape[0]:
            raise ValueError(
                "trusted_empty must have one value per selected Focus crop "
                f"({trusted_empty.numel()} != {ground_truth.shape[0]})"
            )
    if positive_only is None:
        positive_only = torch.zeros_like(lesion_crop, dtype=torch.bool)
    else:
        positive_only = torch.as_tensor(
            positive_only, device=ground_truth.device, dtype=torch.bool
        ).flatten()
        if positive_only.numel() != ground_truth.shape[0]:
            raise ValueError(
                "positive_only must have one value per selected Focus crop "
                f"({positive_only.numel()} != {ground_truth.shape[0]})"
            )
        # The source pseudo mask is guaranteed to contain foreground, but a
        # 4:1 positive/negative crop sampler may legitimately draw a crop that
        # misses it. Such a crop remains unknown cancer background and follows
        # the same frozen-teacher-safe path as any other empty cancer crop.
    mean_components = probability.new_zeros(())

    spacing = (float(args.space_x), float(args.space_y), float(args.space_z))
    margins = tuple(max(0, int(np.ceil(float(args.bbox_context_mm) / value))) for value in spacing)

    if getattr(args, "component_boxes", False):
        trusted, mean_components = component_box_trusted_mask(ground_truth, margins)
    else:
        for batch_index in range(ground_truth.shape[0]):
            coordinates = ground_truth[batch_index].gt(0.5).nonzero(as_tuple=False)
            if coordinates.numel() == 0:
                # Only the explicit no-lesion dataset reaches this path because the
                # cancer datasets use positive-only patch sampling.
                trusted[batch_index] = True
                continue
            lower = coordinates.amin(dim=0)
            upper = coordinates.amax(dim=0) + 1
            slices = []
            for axis, margin in enumerate(margins):
                start = max(0, int(lower[axis].item()) - margin)
                stop = min(ground_truth.shape[axis + 1], int(upper[axis].item()) + margin)
                slices.append(slice(start, stop))
            trusted[(batch_index, *slices)] = True

    # Both box builders historically mark every empty crop as fully trusted.
    # Restore that only for the proxy-negative stream. Cancer-source empty
    # crops start untrusted and can gain safe background voxels from the teacher.
    empty_crop = ~lesion_crop
    trusted[empty_crop] = False
    trusted |= (empty_crop & trusted_empty).view(-1, 1, 1, 1)
    # A pseudo mask supplies trusted foreground, not trusted background. Clear
    # its geometric box before adding teacher-safe negatives below. This keeps
    # uncertain boundary and missed-lesion voxels gradient-free.
    trusted[positive_only] = False

    teacher_consistency = probability.new_zeros(())
    teacher_positive_voxels = probability.new_zeros(())
    teacher_positive_outside_box_voxels = probability.new_zeros(())
    teacher_positive_inside_box_voxels = probability.new_zeros(())
    pseudo_positive_bce = probability.new_zeros(())
    teacher_positive_empty_cancer_voxels = probability.new_zeros(())
    teacher_positive_empty_cancer_crops = probability.new_zeros(())
    empty_pseudo_positive_bce = probability.new_zeros(())
    teacher_safe_negative_voxels = probability.new_zeros(())
    cancer_negative_crops = (empty_crop & (~trusted_empty)).sum()
    if teacher_logits is not None:
        teacher_probability = teacher_logits.softmax(1)[:, 1].detach()
        teacher_safe_scope = (
            (empty_crop & (~trusted_empty)) | positive_only
        )
        teacher_safe_negative = (
            teacher_safe_scope.view(-1, 1, 1, 1)
            & teacher_probability.le(args.pseudo_negative_threshold)
        )
        trusted |= teacher_safe_negative
        teacher_safe_negative_voxels = teacher_safe_negative.sum()
        suspected_unannotated_lesion = (
            lesion_crop.view(-1, 1, 1, 1)
            & ground_truth.lt(0.5)
            & teacher_probability.ge(args.pseudo_positive_threshold)
        )
        # Split before changing `trusted`: only candidates outside the target
        # box represent unannotated secondary-lesion hypotheses. Candidates
        # inside the trusted box are retained for backward-compatible teacher
        # protection/consistency, but the new positive-learning term must not
        # use them to thicken the annotated primary lesion.
        teacher_positive_outside_box = suspected_unannotated_lesion & (~trusted)
        teacher_positive_inside_box = suspected_unannotated_lesion & trusted
        trusted &= ~suspected_unannotated_lesion

        # The first bounding-box experiment only ignored these voxels.  A
        # small consistency term now preserves the successful checkpoint's
        # high-confidence missing-lesion predictions while supervised target
        # voxels continue to drive adaptation.  This remains a soft teacher
        # target rather than a potentially brittle hard pseudo-label.
        teacher_positive_voxels = suspected_unannotated_lesion.sum()
        teacher_positive_outside_box_voxels = teacher_positive_outside_box.sum()
        teacher_positive_inside_box_voxels = teacher_positive_inside_box.sum()
        teacher_consistency = (
            (probability - teacher_probability).square()
            * suspected_unannotated_lesion.float()
        ).sum() / teacher_positive_voxels.clamp_min(1.0)
        # The confidence threshold supplies the precision gate; the hard
        # positive target supplies a directional signal that consistency alone
        # cannot.  Keeping this outside Tversky avoids making a teacher island
        # count like fully annotated disease throughout the crop.
        pseudo_positive_bce = (
            -torch.log(probability.clamp_min(eps))
            * teacher_positive_outside_box.float()
        ).sum() / teacher_positive_outside_box_voxels.clamp_min(1.0)

        # A negative crop sampled from a partially labelled cancer scan may
        # contain a distant metastasis that lies outside every target-centered
        # training crop. Promote it only when the frozen teacher supplies a
        # sizeable, very high-confidence island. Keep this term separate from
        # target-box pseudo positives so its default remains exactly inert.
        if getattr(args, "bbox_empty_pseudo_positive_weight", 0.0) > 0:
            empty_cancer = empty_crop & (~trusted_empty)
            empty_teacher_candidate = (
                empty_cancer.view(-1, 1, 1, 1)
                & teacher_probability.ge(args.pseudo_positive_threshold)
            )
            empty_teacher_counts = empty_teacher_candidate.flatten(1).sum(1)
            qualifying_empty_crop = empty_cancer & empty_teacher_counts.ge(
                int(args.bbox_empty_pseudo_positive_min_voxels)
            )
            teacher_positive_empty_cancer = (
                empty_teacher_candidate
                & qualifying_empty_crop.view(-1, 1, 1, 1)
            )
            teacher_positive_empty_cancer_voxels = teacher_positive_empty_cancer.sum()
            teacher_positive_empty_cancer_crops = qualifying_empty_crop.sum()
            # Thresholds make this disjoint from teacher-safe background, but
            # remove it explicitly from trusted negatives as an invariant.
            trusted &= ~teacher_positive_empty_cancer
            empty_pseudo_positive_bce = (
                -torch.log(probability.clamp_min(eps))
                * teacher_positive_empty_cancer.float()
            ).sum() / teacher_positive_empty_cancer_voxels.clamp_min(1.0)

    # The provided target is always trusted, including any foreground voxels
    # close to a crop boundary.
    trusted |= ground_truth.gt(0.5)
    trusted_float = trusted.float()

    reduce_dims = (1, 2, 3)
    tp = (probability * ground_truth * trusted_float).sum(reduce_dims)
    fp = (probability * (1.0 - ground_truth) * trusted_float).sum(reduce_dims)
    fn = ((1.0 - probability) * ground_truth * trusted_float).sum(reduce_dims)
    tversky = 1.0 - ((tp + eps) / (
        tp + args.tversky_alpha * fp + args.tversky_beta * fn + eps
    )).mean()

    voxel_bce = F.binary_cross_entropy(
        probability.clamp(eps, 1.0 - eps), ground_truth, reduction="none"
    )
    masked_bce = (voxel_bce * trusted_float).sum() / trusted_float.sum().clamp_min(1.0)

    gt_5d = ground_truth.unsqueeze(1)
    dilated = F.max_pool3d(gt_5d, kernel_size=(5, 5, 1), stride=1, padding=(2, 2, 0))
    eroded = -F.max_pool3d(-gt_5d, kernel_size=(5, 5, 1), stride=1, padding=(2, 2, 0))
    boundary_band = (dilated - eroded).gt(0).squeeze(1) & trusted
    boundary_float = boundary_band.float()
    boundary = (voxel_bce * boundary_float).sum() / boundary_float.sum().clamp_min(1.0)

    total = (
        0.55 * tversky
        + 0.30 * masked_bce
        + args.boundary_weight * boundary
        + args.teacher_consistency_weight * teacher_consistency
        + getattr(args, "bbox_pseudo_positive_weight", 0.0) * pseudo_positive_bce
        + getattr(args, "bbox_empty_pseudo_positive_weight", 0.0)
        * empty_pseudo_positive_bce
    )
    return total, {
        "tversky": tversky.detach(),
        "masked_bce": masked_bce.detach(),
        "boundary": boundary.detach(),
        "teacher_consistency": teacher_consistency.detach(),
        "teacher_positive_voxels": teacher_positive_voxels.detach(),
        "teacher_positive_outside_box_voxels": teacher_positive_outside_box_voxels.detach(),
        "teacher_positive_inside_box_voxels": teacher_positive_inside_box_voxels.detach(),
        "pseudo_positive_bce": pseudo_positive_bce.detach(),
        "empty_pseudo_positive_bce": empty_pseudo_positive_bce.detach(),
        "teacher_positive_empty_cancer_voxels": teacher_positive_empty_cancer_voxels.detach(),
        "teacher_positive_empty_cancer_crops": teacher_positive_empty_cancer_crops.detach(),
        "trusted_fraction": trusted_float.mean().detach(),
        "components_per_crop": mean_components.detach(),
        "cancer_negative_crops": cancer_negative_crops.detach(),
        "teacher_safe_negative_voxels": teacher_safe_negative_voxels.detach(),
        "positive_only_crops": positive_only.sum().detach(),
    }


def hybrid_cohort_label_loss(
    student_logits,
    labels,
    complete_label,
    full_loss_func,
    args,
    teacher_logits=None,
    trusted_empty=None,
    positive_only=None,
):
    """Mix complete-mask and partial-label-safe losses within one batch.

    Only cohorts declared in a versioned completeness manifest receive the
    ordinary full-mask DiceCE objective.  Every other cancer crop retains the
    established bounding-box/teacher-safe objective, avoiding false-negative
    gradients on lesions that may be present but unannotated.  Losses are
    weighted by their crop counts so switching batch composition does not
    silently change the optimizer scale.
    """
    complete = torch.as_tensor(
        complete_label, device=student_logits.device, dtype=torch.bool
    ).flatten()
    if complete.numel() != student_logits.shape[0]:
        raise ValueError(
            "complete_label must have one value per selected Focus crop "
            f"({complete.numel()} != {student_logits.shape[0]})"
        )
    partial = ~complete
    complete_count = int(complete.sum().item())
    partial_count = int(partial.sum().item())
    total_count = max(1, complete_count + partial_count)
    zero = student_logits.sum() * 0.0
    full_loss = (
        full_loss_func(student_logits[complete], labels[complete])
        if complete_count else zero
    )
    partial_metrics = {}
    if partial_count:
        partial_teacher = teacher_logits[partial] if teacher_logits is not None else None
        partial_trusted_empty = (
            torch.as_tensor(
                trusted_empty, device=student_logits.device, dtype=torch.bool
            ).flatten()[partial]
            if trusted_empty is not None else None
        )
        partial_positive_only = (
            torch.as_tensor(
                positive_only, device=student_logits.device, dtype=torch.bool
            ).flatten()[partial]
            if positive_only is not None else None
        )
        partial_loss, partial_metrics = bbox_partial_label_loss(
            student_logits[partial],
            labels[partial],
            args,
            teacher_logits=partial_teacher,
            trusted_empty=partial_trusted_empty,
            positive_only=partial_positive_only,
        )
    else:
        partial_loss = zero
    total = (
        full_loss * (complete_count / total_count)
        + partial_loss * (partial_count / total_count)
    )
    metrics = dict(partial_metrics)
    metrics.update(
        {
            "hybrid_complete_crops": student_logits.new_tensor(complete_count),
            "hybrid_partial_crops": student_logits.new_tensor(partial_count),
            "hybrid_full_loss": full_loss.detach(),
            "hybrid_partial_loss": partial_loss.detach(),
        }
    )
    return total, metrics


def train_epoch(model, loader, optimizers, scheduler, scaler, epoch, loss_func, args, old_policy_model,
                teacher_segmentation=None):
    model.train()
    core_model = model.module if args.distributed else model
    # requires_grad=False freezes parameters but not BatchNorm running buffers.
    # Keep Glance in eval mode during Focus-only adaptation so crop selection is
    # truly identical to the source checkpoint.
    if getattr(args, "freeze_glance", False):
        core_model.glance_model.eval()
    # Likewise, Glance-only adaptation must not update stochastic behavior or
    # normalization buffers in the frozen Focus network.
    if getattr(args, "freeze_focus", False):
        core_model.segmentation_model.eval()
    start_time = time.time()
    run_loss = AverageMeter()
    partial_metric_sums = {}
    partial_metric_steps = 0

    optimizer_seg, optimizer_glance = optimizers

    old_policy_model.eval()

    for idx, batch_data in enumerate(loader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        trusted_empty = (
            batch_data.get("trusted_empty")
            if isinstance(batch_data, dict) else None
        )
        complete_label = (
            batch_data.get("complete_label")
            if isinstance(batch_data, dict) else None
        )
        positive_only_label = (
            batch_data.get("positive_only_label")
            if isinstance(batch_data, dict) else None
        )
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        if optimizer_seg is not None:
            optimizer_seg.zero_grad()
        if optimizer_glance is not None:
            optimizer_glance.zero_grad()

        partial_metrics = {}
        box_loss = data.sum() * 0.0
        center_loss = data.sum() * 0.0
        glance_distillation_loss = data.sum() * 0.0
        classification_only = (
            getattr(args, "freeze_focus", False)
            and float(getattr(args, "glance_rl_weight", 1.0)) == 0.0
        )
        if classification_only:
            if getattr(args, "glance_teacher_labels", False):
                glance_output = model(
                    data,
                    target,
                    old_policy_model,
                    return_crops=True,
                    classification_only=True,
                )
                auxiliary_localization = (
                    getattr(args, "glance_box_localization", False)
                    or getattr(args, "glance_center_localization", False)
                )
                if auxiliary_localization:
                    (
                        actions,
                        labels,
                        gl_loss,
                        glance_crops,
                        glance_crop_labels,
                        glance_box_logits,
                        glance_center_logits,
                    ) = glance_output
                else:
                    actions, labels, gl_loss, glance_crops = glance_output
                    glance_crop_labels = None
                    glance_box_logits = None
                    glance_center_logits = None
                if getattr(args, "glance_box_localization", False):
                    box_loss, box_metrics = glance_component_box_loss(
                        glance_box_logits, glance_crop_labels
                    )
                    partial_metrics.update(box_metrics)
                else:
                    box_loss = gl_loss.new_zeros(())
                if getattr(args, "glance_center_localization", False):
                    center_loss, center_metrics = glance_component_center_loss(
                        glance_center_logits,
                        glance_crop_labels,
                        complete_label=complete_label
                        if getattr(args, "hybrid_cohort_glance_labels", False)
                        else None,
                    )
                    partial_metrics.update(center_metrics)
                else:
                    center_loss = gl_loss.new_zeros(())
                teacher_outputs = []
                teacher_batch_size = int(getattr(args, "glance_teacher_batch_size", 2))
                teacher_network = (
                    teacher_segmentation
                    if teacher_segmentation is not None
                    else core_model.segmentation_model
                )
                with torch.no_grad(), autocast(enabled=args.amp):
                    for start in range(0, glance_crops.shape[0], teacher_batch_size):
                        teacher_outputs.append(
                            teacher_network(
                                glance_crops[start:start + teacher_batch_size]
                            )
                        )
                teacher_actions = torch.cat(teacher_outputs, dim=0)
                gl_loss, glance_metrics = teacher_safe_glance_loss(
                    actions,
                    labels,
                    teacher_actions,
                    pseudo_positive_threshold=args.glance_pseudo_positive_threshold,
                    pseudo_negative_threshold=args.glance_pseudo_negative_threshold,
                    pseudo_positive_min_voxels=args.glance_pseudo_positive_min_voxels,
                    complete_label=complete_label
                    if getattr(args, "hybrid_cohort_glance_labels", False)
                    else None,
                )
                partial_metrics.update(glance_metrics)
                if getattr(args, "glance_distillation_weight", 0.0) > 0:
                    with torch.no_grad(), autocast(enabled=args.amp):
                        reference_actions = old_policy_model.get_actions(glance_crops)
                    glance_distillation_loss, distillation_metrics = (
                        glance_ignored_distillation_loss(
                            actions,
                            reference_actions,
                            labels,
                            teacher_actions,
                            pseudo_positive_threshold=args.glance_pseudo_positive_threshold,
                            pseudo_negative_threshold=args.glance_pseudo_negative_threshold,
                            pseudo_positive_min_voxels=args.glance_pseudo_positive_min_voxels,
                            complete_label=complete_label
                            if getattr(args, "hybrid_cohort_glance_labels", False)
                            else None,
                        )
                    )
                    partial_metrics.update(distillation_metrics)
            else:
                actions, labels, gl_loss = model(
                    data, target, old_policy_model, classification_only=True
                )
                box_loss = gl_loss.new_zeros(())
                center_loss = gl_loss.new_zeros(())
            seg_loss = gl_loss.new_zeros(())
            RL_loss = gl_loss.new_zeros(())
            seg = seg_labels = None
        elif teacher_segmentation is not None or getattr(args, "glance_teacher_labels", False):
            seg, seg_labels, actions, labels, gl_loss, RL_loss, seg_crops, selected_indices = model(
                data, target, old_policy_model, True
            )
            selected_trusted_empty = (
                trusted_empty.flatten()[selected_indices].to(device=seg.device)
                if trusted_empty is not None else None
            )
            selected_complete_label = (
                complete_label.flatten()[selected_indices].to(device=seg.device)
                if complete_label is not None else None
            )
            selected_positive_only = (
                positive_only_label.flatten()[selected_indices].to(device=seg.device)
                if positive_only_label is not None else None
            )
            if getattr(args, "glance_teacher_labels", False) and getattr(args, "freeze_focus", False):
                # Focus is frozen and already ran for the GRL reward. Reuse
                # those exact logits as crop-label teacher predictions.
                teacher_seg = seg.detach()
            else:
                with torch.no_grad(), autocast(enabled=args.amp):
                    teacher_seg = teacher_segmentation(seg_crops)
        else:
            seg, seg_labels, actions, labels, gl_loss, RL_loss = model(data, target, old_policy_model)

        if getattr(args, "glance_teacher_labels", False) and not classification_only:
            gl_loss, glance_metrics = teacher_safe_glance_loss(
                actions[selected_indices],
                labels[selected_indices],
                teacher_seg,
                pseudo_positive_threshold=args.glance_pseudo_positive_threshold,
                pseudo_negative_threshold=args.glance_pseudo_negative_threshold,
                pseudo_positive_min_voxels=args.glance_pseudo_positive_min_voxels,
                complete_label=selected_complete_label
                if getattr(args, "hybrid_cohort_glance_labels", False)
                else None,
            )
            partial_metrics.update(glance_metrics)

        # segmentation loss
        if classification_only or getattr(args, "freeze_focus", False):
            if not classification_only:
                seg_loss = seg.sum() * 0.0
            pass
        elif getattr(args, "hybrid_cohort_label_loss", False):
            if selected_complete_label is None:
                raise RuntimeError(
                    "hybrid cohort loss requires complete_label metadata"
                )
            seg_loss, partial_metrics = hybrid_cohort_label_loss(
                seg,
                seg_labels,
                selected_complete_label,
                loss_func,
                args,
                teacher_logits=teacher_seg,
                trusted_empty=selected_trusted_empty,
                positive_only=selected_positive_only,
            )
        elif args.bbox_partial_label:
            seg_loss, partial_metrics = bbox_partial_label_loss(
                seg, seg_labels, args, teacher_logits=teacher_seg
                if teacher_segmentation is not None else None,
                trusted_empty=selected_trusted_empty
                if teacher_segmentation is not None else None,
                positive_only=selected_positive_only,
            )
        elif teacher_segmentation is not None:
            seg_loss, partial_metrics = partial_label_teacher_loss(seg, seg_labels, teacher_seg, args)
        else:
            seg_loss = loss_func(seg, seg_labels)
        if partial_metrics:
            partial_metric_steps += 1
            for key, value in partial_metrics.items():
                partial_metric_sums[key] = partial_metric_sums.get(key, 0.0) + float(
                    value.float().mean().item()
                )
        # Keep the released recipe as the default, but expose both Glance
        # objectives.  The RL reward is zero when Focus misses a real lesion;
        # over-weighting it can therefore teach Glance to suppress precisely
        # the hard positive crops we need for high-recall segmentation.
        total_loss = (
            seg_loss
            + gl_loss * args.glance_classification_weight
            + RL_loss * args.glance_rl_weight
            + box_loss * getattr(args, "glance_box_localization_weight", 0.0)
            + center_loss
            * getattr(args, "glance_center_localization_weight", 0.0)
            + glance_distillation_loss
            * getattr(args, "glance_distillation_weight", 0.0)
        )

        if args.amp:
            scaler.scale(total_loss).backward()
            if optimizer_seg is not None:
                scaler.step(optimizer_seg)
            if optimizer_glance is not None:
                scaler.step(optimizer_glance)
            scaler.update()
        else:
            total_loss.backward()
            if optimizer_seg is not None:
                optimizer_seg.step()
            if optimizer_glance is not None:
                optimizer_glance.step()

        run_loss.update(total_loss.item(), n=args.batch_size)

        lr = (
            optimizer_seg.param_groups[0]["lr"]
            if optimizer_seg is not None
            else optimizer_glance.param_groups[0]["lr"]
        )
        if scheduler is not None:
            scheduler_seg, scheduler_glance = scheduler
            if scheduler_seg is not None:
                scheduler_seg.step()
            if scheduler_glance is not None:
                scheduler_glance.step()

        length = max(1, len(loader) // 4)
        if args.rank == 0 and (idx + 1) % length == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "lr: {:.8f}".format(lr),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    partial_metric_means = {
        key: value / max(1, partial_metric_steps)
        for key, value in partial_metric_sums.items()
    }
    return run_loss.avg, (optimizer_seg, optimizer_glance), partial_metric_means


def val_epoch(model, loader, epoch, acc_func, args, post_label=None, post_pred=None):
    model.eval()
    core_model = model.module if args.distributed else model
    run_acc = AverageMeter()
    recall_counts = {
        "annotated_voxel_tp": 0,
        "annotated_voxel_total": 0,
        "annotated_components_detected": 0,
        "annotated_components_total": 0,
    }
    for tag, _lower, _upper in ANNOTATED_RECALL_BINS_MM3:
        recall_counts[f"annotated_components_{tag}_detected"] = 0
        recall_counts[f"annotated_components_{tag}_total"] = 0
    start_time = time.time()
    with torch.no_grad():

        metric_count = 0
        for idx, batch_data in enumerate(loader):
            torch.cuda.empty_cache()

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                seg, output_labels, actions = core_model.valid_with_labels(data, target)

            # For Glance
            # For Segmentation, target for segmentation evaluation
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(seg)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            valid_for_diagnostics = (
                not args.distributed
                or idx < getattr(loader.sampler, "valid_length", len(loader))
            )
            if valid_for_diagnostics:
                case_counts = annotated_recall_counts(
                    val_output_convert[0][1].detach().cpu().numpy() > 0.5,
                    val_labels_convert[0][1].detach().cpu().numpy() > 0.5,
                    voxel_mm3=(
                        float(args.space_x)
                        * float(args.space_y)
                        * float(args.space_z)
                    ),
                )
                for key, value in case_counts.items():
                    recall_counts[key] += int(value)
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            seg_acc, not_nans = acc_func.aggregate()
            seg_acc = seg_acc.cuda(args.rank)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [seg_acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(seg_acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                seg_dice = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "seg_dice: {:.4f}".format(seg_dice),
                    "time {:.2f}s".format(time.time() - start_time),
            )
            start_time = time.time()

    count_keys = sorted(recall_counts)
    count_tensor = torch.as_tensor(
        [recall_counts[key] for key in count_keys],
        dtype=torch.float64,
        device=torch.device("cuda", args.rank),
    )
    if args.distributed:
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
    global_counts = {
        key: int(round(value))
        for key, value in zip(count_keys, count_tensor.cpu().tolist())
    }
    diagnostics = {
        "annotated_voxel_recall": (
            global_counts["annotated_voxel_tp"]
            / max(1, global_counts["annotated_voxel_total"])
        ),
        "annotated_component_recall": (
            global_counts["annotated_components_detected"]
            / max(1, global_counts["annotated_components_total"])
        ),
        **global_counts,
    }
    for tag, _lower, _upper in ANNOTATED_RECALL_BINS_MM3:
        diagnostics[f"annotated_component_recall_{tag}"] = (
            global_counts[f"annotated_components_{tag}_detected"]
            / max(1, global_counts[f"annotated_components_{tag}_total"])
        )
    val_epoch.last_diagnostics = diagnostics
    if args.rank == 0:
        print(
            "Annotated-recall diagnostics",
            " ".join(
                f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in sorted(diagnostics.items())
            ),
        )
    return run_acc.avg


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizers=None, schedulers=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizers is not None:
        if optimizers[0] is not None:
            save_dict["optimizer_seg"] = optimizers[0].state_dict()
        if optimizers[1] is not None:
            save_dict["optimizer_glance"] = optimizers[1].state_dict()
    if schedulers is not None:
        if schedulers[0] is not None:
            save_dict["scheduler_seg"] = schedulers[0].state_dict()
        if schedulers[1] is not None:
            save_dict["scheduler_glance"] = schedulers[1].state_dict()
    filename = os.path.join(args.logdir, filename)
    atomic_torch_save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizers,
        loss_func,
        acc_func,
        args,
        scheduler=None,
        start_epoch=0,
    post_label=None,
    post_pred=None,
    teacher_segmentation=None,
    best_acc=0.0,
):
    from models.glance import Glance_model
    old_policy_model = Glance_model(
        args.glance_backbone,
        box_localization=getattr(args, "glance_box_localization", False),
        center_localization=getattr(args, "glance_center_localization", False),
    )
    old_policy_model.cuda(args.gpu)
    core_model = model.module if args.distributed else model
    old_policy_model.load_state_dict(core_model.glance_model.state_dict(), strict=True)
    old_policy_model.eval()

    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    # Preserve the best validation score across an exact checkpoint resume so
    # a worse post-resume epoch cannot replace model.pt.
    val_acc_max = float(best_acc)
    optimizer_seg, optimizer_glance = optimizers

    wandb_run = None
    if args.rank == 0 and args.wandb_mode != "disabled":
        wandb_id = stable_wandb_run_id(args.logdir, args.wandb_name)
        wandb_run = safe_wandb_init(
            project=args.wandb_project,
            name=args.wandb_name,
            id=wandb_id,
            dir=args.logdir,
            mode=args.wandb_mode,
            config=vars(args),
            resume="allow",
        )
        wandb_run = safe_wandb_define_metrics(wandb_run, [
            ("epoch", {}),
            ("train/*", {"step_metric": "epoch"}),
            ("val/*", {"step_metric": "epoch"}),
        ])
        print(f"W&B stable run id: {wandb_id}", flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        if epoch % args.update_epochs == 0:
            old_policy_model.load_state_dict(core_model.glance_model.state_dict(), strict=True)
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if args.distributed:
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()

        train_loss, (optimizer_seg, optimizer_glance), partial_metric_means = train_epoch(
            model, train_loader, (optimizer_seg, optimizer_glance), scheduler, scaler=scaler,
            epoch=epoch, loss_func=loss_func, args=args, old_policy_model=old_policy_model,
            teacher_segmentation=teacher_segmentation,
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            if partial_metric_means:
                print(
                    "Partial-label metrics",
                    " ".join(
                        f"{key}={value:.6f}"
                        for key, value in sorted(partial_metric_means.items())
                    ),
                )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        if wandb_run is not None:
            train_log = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/lr_focus": optimizer_seg.param_groups[0]["lr"] if optimizer_seg is not None else 0.0,
                "train/lr_glance": optimizer_glance.param_groups[0]["lr"] if optimizer_glance is not None else 0.0,
            }
            train_log.update({
                f"train/{key}": value
                for key, value in partial_metric_means.items()
            })
            wandb_run = safe_wandb_log(
                wandb_run, train_log, step=epoch + 1, commit=False
            )
        b_new_best = False

        checkpoint_every = int(getattr(args, "checkpoint_every", 1))
        retain_epoch = (
            (epoch + 1) % checkpoint_every == 0
            or (epoch + 1) == args.max_epochs
        )
        if args.rank == 0 and args.logdir is not None and retain_epoch:
            # Every retained state is immutable.  The opt-in interval prevents
            # long specialist runs from exhausting nearly-full local storage;
            # the default remains one checkpoint per epoch.
            save_checkpoint(
                model, epoch, args, best_acc=val_acc_max,
                filename=f"checkpoint_epoch_{epoch + 1:04d}.pt",
                optimizers=(optimizer_seg, optimizer_glance), schedulers=scheduler,
            )

        # Validate the final epoch unless an external checkpoint-grid
        # controller explicitly owns model selection. Every epoch is saved
        # above, so deferred validation does not reduce recoverability.
        should_validate = (
            (epoch + 1) % args.val_every == 0
            or (
                (epoch + 1) == args.max_epochs
                and not getattr(args, "skip_final_validation", False)
            )
        )
        if should_validate:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            val_diagnostics = dict(
                getattr(val_epoch, "last_diagnostics", {})
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "dice",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                    for key, value in val_diagnostics.items():
                        writer.add_scalar(f"val_{key}", value, epoch)
                if wandb_run is not None:
                    validation_log = {
                        "epoch": epoch + 1,
                        "val/dice": val_avg_acc,
                    }
                    validation_log.update(
                        {
                            f"val/{key}": value
                            for key, value in val_diagnostics.items()
                        }
                    )
                    wandb_run = safe_wandb_log(
                        wandb_run,
                        validation_log,
                        step=epoch + 1,
                    )
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max,
                            optimizers=(optimizer_seg, optimizer_glance), schedulers=scheduler,
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(
                    model, epoch, args, best_acc=val_acc_max, filename="model_final.pt",
                    optimizers=(optimizer_seg, optimizer_glance), schedulers=scheduler,
                )
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    atomic_copyfile(
                        os.path.join(args.logdir, "model_final.pt"),
                        os.path.join(args.logdir, "model.pt"),
                    )

        if wandb_run is not None and not should_validate:
            wandb_run = safe_wandb_log(
                wandb_run, {"epoch": epoch + 1}, step=epoch + 1
            )

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    if writer is not None:
        writer.close()
    safe_wandb_finish(wandb_run)

    return val_acc_max
