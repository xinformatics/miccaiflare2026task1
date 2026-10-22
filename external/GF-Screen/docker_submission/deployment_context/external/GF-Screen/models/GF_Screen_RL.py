from models.glance import (
    DEFAULT_CROP_SIZE,
    Glance_model,
    recenter_crops_3d_tensor,
    select_crops_from_proposal,
)
from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn
import torch.nn.functional as F

def select_focus_crops(
    actions,
    glance_threshold,
    proposal_selected=None,
    proposal_only=False,
):
    """Combine Glance and optional external-proposal crop selections.

    ``proposal_only`` is useful for proposal-source ablations and oracle
    diagnostics: it cleanly measures Focus performance without retaining any
    Glance-selected crop. Normal inference remains the exact logical union of
    Glance and proposal selections.
    """
    selected = actions.softmax(1)[:, 1] > glance_threshold
    if proposal_only:
        if proposal_selected is None:
            raise ValueError("proposal_only requires proposal crop selections")
        selected = torch.zeros_like(selected)
    if proposal_selected is not None:
        if proposal_selected.shape != selected.shape:
            raise ValueError(
                f"proposal/action selection shape mismatch: "
                f"{proposal_selected.shape} versus {selected.shape}"
            )
        selected = selected | proposal_selected.to(device=selected.device, dtype=torch.bool)
    return selected


def select_center_crops(center_logits, threshold):
    """Select crops whose dense centre head contains a confident lesion peak."""
    if center_logits.ndim != 5 or center_logits.shape[1] != 1:
        raise ValueError(
            "center logits must have shape (crops,1,D,H,W), found "
            f"{tuple(center_logits.shape)}"
        )
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("center threshold must lie strictly between zero and one")
    return center_logits.sigmoid().flatten(1).amax(1).ge(float(threshold))


def focus_probability_with_tta(segmentation_model, batch, flip_dims=()):
    """Average Focus probabilities over optional single-axis flip views.

    ``batch`` has spatial dimensions 2, 3, and 4. Each augmented prediction is
    flipped back before averaging. An empty ``flip_dims`` tuple is exactly the
    legacy single-view softmax path.
    """
    probability = segmentation_model(batch).softmax(1)
    for dimension in flip_dims:
        flipped_batch = torch.flip(batch, dims=(dimension,))
        flipped_probability = segmentation_model(flipped_batch).softmax(1)
        probability = probability + torch.flip(flipped_probability, dims=(dimension,))
    return probability / (1 + len(flip_dims))


def teacher_safe_glance_masks(
    annotated_labels,
    teacher_logits,
    pseudo_positive_threshold=0.95,
    pseudo_negative_threshold=0.05,
    pseudo_positive_min_voxels=64,
):
    """Partition crops into safe positive, negative, and ignored targets."""
    if not 0.0 <= pseudo_negative_threshold < pseudo_positive_threshold <= 1.0:
        raise ValueError("Glance teacher thresholds must satisfy 0 <= negative < positive <= 1")
    if pseudo_positive_min_voxels < 1:
        raise ValueError("Glance pseudo-positive minimum voxels must be at least 1")
    if annotated_labels.shape[0] != teacher_logits.shape[0]:
        raise ValueError("Glance labels and teacher logits must have matching batches")

    if annotated_labels.ndim > 2:
        annotated_positive = annotated_labels.flatten(1).gt(0).any(1)
    else:
        annotated_positive = annotated_labels.flatten().bool()
    teacher_probability = teacher_logits.softmax(1)[:, 1].detach()
    teacher_positive_voxels = teacher_probability.ge(
        pseudo_positive_threshold
    ).flatten(1).sum(1)
    pseudo_positive = (~annotated_positive) & teacher_positive_voxels.ge(
        int(pseudo_positive_min_voxels)
    )
    safe_negative = (~annotated_positive) & teacher_probability.flatten(1).amax(1).le(
        pseudo_negative_threshold
    )
    positive = annotated_positive | pseudo_positive
    valid = positive | safe_negative
    return (
        annotated_positive,
        pseudo_positive,
        safe_negative,
        valid,
        teacher_positive_voxels,
    )


class GFScreen_RL(nn.Module):
    def __init__(self, args):
        super(GFScreen_RL, self).__init__()
        self.glance_model = Glance_model(
            backbone=args.glance_backbone,
            grid_mode=getattr(args, "grid_mode", "legacy"),
            box_localization=getattr(args, "glance_box_localization", False),
            center_localization=getattr(args, "glance_center_localization", False),
        )
        # MONAI >= 1.5 infers the input size dynamically and removed img_size.
        # The released checkpoint is otherwise shape-compatible with use_v2=True.
        swin_kwargs = dict(in_channels=args.in_channels,
                           out_channels=args.out_channels,
                           feature_size=args.feature_size,
                           use_checkpoint=args.use_checkpoint,
                           use_v2=True)
        try:
            # MONAI >= 1.5 infers the input size dynamically.
            self.segmentation_model = SwinUNETR(**swin_kwargs)
        except TypeError as error:
            # The CUDA-11-compatible challenge image uses an older MONAI whose
            # SwinUNETR constructor still requires img_size. The argument does
            # not alter checkpoint tensor shapes.
            if "img_size" not in str(error):
                raise
            self.segmentation_model = SwinUNETR(
                img_size=(args.roi_x, args.roi_y, args.roi_z), **swin_kwargs
            )

        self.num_crops = args.batch_size * args.sw_batch_size
        self.args = args

    def get_RL_loss(self, actions_logits, old_policy_model, seg_crops, seg, seg_labels):
        # GRPO reinforcement training for Glance and Focus
        # actions_logits: (b, 2) - original logits
        # seg_crops: (b, 1, h, w, d), selected input
        # seg: segmentation predictions, (b, 2, h, w, d)
        # seg_labels: (b, 1, h, w, d), 0 and 1

        # larger temperature, smooth logits
        temperature = 1.0
        actions_softmax = F.softmax(actions_logits / temperature, dim=1)[:, 1].unsqueeze(1)

        # log actions_softmax
        actions_log_prob = torch.log(actions_softmax + 1e-6)

        # get old actions from old policy model, no gradient
        global_valid_count = None
        distributed_world_size = 1
        with torch.no_grad():
            old_actions = old_policy_model.get_actions(seg_crops)
            old_actions_softmax = F.softmax(old_actions / temperature, dim=1)[:, 1].unsqueeze(1)
            old_actions_log_prob = torch.log(old_actions_softmax + 1e-6)
            old_actions_log_prob = old_actions_log_prob.detach()

            # Get reward
            reward = get_detection_reward(pred=seg.detach(), label=seg_labels)
            valid_reward = torch.ones_like(reward, dtype=torch.bool)
            teacher_safe_rewards = getattr(self.args, "glance_teacher_labels", False)
            if teacher_safe_rewards:
                (
                    _annotated_positive,
                    pseudo_positive,
                    _safe_negative,
                    valid_reward,
                    _teacher_positive_voxels,
                ) = teacher_safe_glance_masks(
                    seg_labels,
                    seg.detach(),
                    pseudo_positive_threshold=self.args.glance_pseudo_positive_threshold,
                    pseudo_negative_threshold=self.args.glance_pseudo_negative_threshold,
                    pseudo_positive_min_voxels=self.args.glance_pseudo_positive_min_voxels,
                )
                reward = torch.where(pseudo_positive, torch.ones_like(reward), reward)

            # Group advantage from rewards. advantages shape: (b, 1)
            if (
                getattr(self.args, "global_group_advantages", False)
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                distributed_world_size = torch.distributed.get_world_size()
                gathered_rewards = [torch.zeros_like(reward) for _ in range(distributed_world_size)]
                torch.distributed.all_gather(gathered_rewards, reward)
                global_rewards = torch.cat(gathered_rewards, dim=0)
                if teacher_safe_rewards:
                    gathered_valid = [
                        torch.zeros_like(valid_reward)
                        for _ in range(distributed_world_size)
                    ]
                    torch.distributed.all_gather(gathered_valid, valid_reward)
                    global_valid = torch.cat(gathered_valid, dim=0)
                    global_valid_count = global_valid.float().sum().clamp_min(1.0)
                    global_advantages = masked_group_relative_advantages(
                        rewards=global_rewards, valid=global_valid
                    )
                else:
                    global_valid_count = reward.new_tensor(float(global_rewards.numel()))
                    global_advantages = group_relative_advantages(
                        rewards=global_rewards
                    )
                rank = torch.distributed.get_rank()
                start = rank * reward.numel()
                advantages = global_advantages[start:start + reward.numel()]
            else:
                advantages = (
                    masked_group_relative_advantages(
                        rewards=reward, valid=valid_reward
                    )
                    if teacher_safe_rewards
                    else group_relative_advantages(rewards=reward)
                )
            advantages = advantages.unsqueeze(1)

        # actions_log_prob, old_actions_log_prob: (b, 1)
        ratio = torch.exp(actions_log_prob - old_actions_log_prob)
        ratio = torch.clamp(ratio, 0.01, 5)  # clamp abnormal value

        # GRPO clip
        epsilon = 0.1
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        valid_float = valid_reward.float().unsqueeze(1)
        valid_count = valid_float.sum().clamp_min(1.0)
        # DDP averages gradients across ranks. Scale each local numerator by
        # world_size/global_valid_count so that the averaged gradient equals a
        # true mean over all trusted crops, even when valid counts differ by
        # rank. The legacy/local path retains its exact local mean.
        reduction_scale = (
            float(distributed_world_size) / global_valid_count
            if global_valid_count is not None
            else 1.0 / valid_count
        )
        loss_clip = -(
            torch.min(surr1, surr2) * valid_float
        ).sum() * reduction_scale

        # GRPO KL divergence
        beta = 0.01
        p = old_actions_softmax / (actions_softmax + 1e-6)
        p = torch.clamp(p, 1e-4, 1e4)  # clamp abnormal value
        KL_divergence = p - torch.log(p + 1e-6) - 1
        KL_divergence = beta * (KL_divergence * valid_float).sum() * reduction_scale

        RL_loss = loss_clip + KL_divergence

        return RL_loss

    def forward(
        self,
        input_images,
        input_labels,
        old_policy_model,
        return_crops=False,
        classification_only=False,
    ):
        b, _, x, y, z = input_images.size()

        # cropped images, labels, and actions
        # actions.shape: (crops, 2)
        glance_output = self.glance_model.forward_with_labels(
            input_images,
            input_labels,
            training=True,
            return_box_predictions=(
                classification_only
                and getattr(self.args, "glance_box_localization", False)
            ),
            return_center_predictions=(
                classification_only
                and getattr(self.args, "glance_center_localization", False)
            ),
        )
        auxiliary_localization = classification_only and (
            getattr(self.args, "glance_box_localization", False)
            or getattr(self.args, "glance_center_localization", False)
        )
        if not auxiliary_localization:
            crops, coords_all, labels, actions = glance_output
            box_predictions = center_predictions = None
        else:
            crops, coords_all, labels, actions, *auxiliary = glance_output
            auxiliary_index = 0
            box_predictions = None
            center_predictions = None
            if getattr(self.args, "glance_box_localization", False):
                box_predictions = auxiliary[auxiliary_index]
                auxiliary_index += 1
            if getattr(self.args, "glance_center_localization", False):
                center_predictions = auxiliary[auxiliary_index]

        if classification_only:
            glance_labels = get_glance_label(labels)
            glance_loss = get_glance_loss(actions, glance_labels.squeeze(1))
            if return_crops:
                # Teacher-safe Glance adaptation needs the exact random crops
                # corresponding to every action.  Preserve the legacy
                # three-value return unless explicitly requested.
                output = (actions, glance_labels, glance_loss, crops)
                if auxiliary_localization:
                    return output + (
                        labels,
                        box_predictions,
                        center_predictions,
                    )
                return output
            return actions, glance_labels, glance_loss

        # random indices
        indices = torch.randperm(b)

        # actions for RL loss
        select_actions_logits = actions[indices][:self.num_crops]

        # segmentation
        seg_crops = crops[indices][:self.num_crops]
        seg_labels = labels[indices][:self.num_crops]
        seg = self.segmentation_model(seg_crops)

        # get glance loss
        labels = get_glance_label(labels)
        glance_loss = get_glance_loss(actions, labels.squeeze(1))

        # get RL loss
        RL_loss = self.get_RL_loss(actions_logits=select_actions_logits, old_policy_model=old_policy_model,
                                   seg_crops=seg_crops, seg=seg, seg_labels=seg_labels)

        outputs = (seg, seg_labels, actions, labels, glance_loss, RL_loss)
        if return_crops:
            # Return the exact shuffled source indices so source-level trust
            # metadata remains aligned with the selected Focus crops.
            return outputs + (seg_crops, indices[:self.num_crops])
        return outputs

    def valid_with_labels(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # Validation uses the full-volume target only for the final Dice score.
        # Materializing a 96x96x64 label tensor for every Glance crop is unused
        # downstream and can exceed 10 GiB on large scans. Use the label-free
        # inference path and keep the target as one full-volume tensor instead.
        crops, coords_all, actions = self.glance_model(input_images)
        labels = None

        # select by actions
        actions_argmax = actions.argmax(1)

        if actions_argmax.sum() > 0:
            selected_index = (actions_argmax == 1)

            # Keep only integer crop ids, not a second full selected-crop
            # tensor. Aggregate each raw-logit microbatch immediately so an
            # extreme full-body scan cannot retain hundreds of Focus outputs
            # until the end. This is mathematically the same ordered uniform
            # overlap aggregation used by the former concatenate-then-reduce
            # implementation; tests cover exact streamed/full equivalence.
            coord = torch.tensor(coords_all[0], device=input_images.device)
            selected_ids = selected_index.nonzero(as_tuple=False).flatten()
            infer_batch = int(getattr(self.args, "validation_focus_batch_size", 2))
            aggregation_mode = getattr(self.args, "aggregation_mode", "uniform")
            seg_sum = torch.zeros(
                [b, self.args.out_channels, x, y, z], device=input_images.device
            )
            count_map = torch.zeros_like(seg_sum)
            gaussian_map = None
            for i in range(0, selected_ids.numel(), infer_batch):
                batch_ids = selected_ids[i:i + infer_batch]
                output = self.segmentation_model(crops[batch_ids])
                gaussian_map = accumulate_segmentation_with_coords(
                    seg_sum,
                    count_map,
                    output,
                    coord[batch_ids],
                    mode=aggregation_mode,
                    gaussian_map=gaussian_map,
                )
            seg_out = finalize_segmentation_aggregation(seg_sum, count_map)

        else:
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        # seg_out: segmentation for the whole volume
        # labels: for glance (num_crops, 2)
        # actions: (num_crops, 2)
        return seg_out, labels, actions

    def view_features(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels, training=False)

        # inference with batch, to avoid large batch for segmentation --> out of memory
        infer_batch = 4
        total_samples = crops.size()[0]

        seg = []
        features = []
        for i in range(0, total_samples, infer_batch):
            batch = crops[i:i + infer_batch]
            output,feature = self.segmentation_model(batch)
            seg.append(output)
            features.append(feature)
        # Concatenate all results
        seg = torch.cat(seg, dim=0)
        features = torch.cat(features, dim=0)

        return crops, seg, labels, actions, features

    def view_optimal(self, input_images, input_labels):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        # actions.shape: (crops, 2)
        crops, coords_all, labels, actions = self.glance_model.forward_with_labels(input_images, input_labels, training=False)

        # select by actions
        actions_argmax = actions.argmax(1)

        if actions_argmax.sum() > 0:
            selected_index = (actions_argmax == 1)
            # print('selected_index:', selected_index)

            selected_crops = crops[selected_index]
            # print('crops.shape, selected_crops.shape: ', crops.shape, selected_crops.shape)

            # Convert coord to a tensor first
            coord = torch.tensor(coords_all[0]).to(input_images.device)
            selected_coord = coord[selected_index]

            # inference with batch, to avoid large batch for segmentation --> out of memory
            infer_batch = getattr(self.args, "focus_batch_size", 4)
            total_samples = selected_crops.size()[0]
            seg = []
            for i in range(0, total_samples, infer_batch):
                batch = selected_crops[i:i + infer_batch]
                output = self.segmentation_model(batch)
                seg.append(output)
            # Concatenate all results
            seg = torch.cat(seg, dim=0)

            zeros = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)
            seg_out = aggregate_segmentation_with_coords(
                zeros,
                seg,
                selected_coord,
                mode=getattr(self.args, "aggregation_mode", "uniform"),
            )

        else:
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        # seg_out: segmentation for the whole volume
        # labels: for glance (num_crops, 2)
        # actions: (num_crops, 2)
        return selected_crops, seg, seg_out, labels, actions

    def valid(self, input_images, proposal_mask=None):
        b, _, x, y, z = input_images.size()
        assert b == 1, print('validation, batch_size should be 1, current batch_size is: ', b)

        center_threshold = getattr(self.args, "glance_center_threshold", None)
        if center_threshold is None:
            crops, coords_all, actions = self.glance_model(input_images)
            center_selected = None
        else:
            if not getattr(self.args, "glance_center_localization", False):
                raise ValueError(
                    "glance_center_threshold requires a center-localization checkpoint"
                )
            crops, coords_all, actions, center_logits = self.glance_model(
                input_images, return_center_predictions=True
            )
            center_selected = select_center_crops(center_logits, center_threshold)

        # select by actions
        glance_threshold = getattr(self.args, "glance_threshold", 0.9)
        proposal_selected = None
        if proposal_mask is not None:
            if proposal_mask.shape != input_images.shape:
                raise ValueError(
                    f"proposal/image shape mismatch: {proposal_mask.shape} versus {input_images.shape}"
                )
            if b != 1:
                raise ValueError("proposal-guided inference currently requires batch size 1")
            proposal_selected = select_crops_from_proposal(
                proposal_mask[0],
                coords_all[0],
                min_voxels=getattr(self.args, "proposal_min_voxels", 1),
            ).to(input_images.device)
        selected_index = select_focus_crops(
            actions,
            glance_threshold,
            proposal_selected=proposal_selected,
            proposal_only=getattr(self.args, "proposal_only", False),
        )
        if center_selected is not None:
            if center_selected.shape != selected_index.shape:
                raise ValueError("center/action crop-selection shape mismatch")
            selected_index = selected_index | center_selected

        selected_count = int(selected_index.sum().item())
        if selected_count > 0:
            compress_ratio = selected_count / crops.shape[0]

            # Do not materialize a second copy of every selected crop or keep
            # every Focus output until the end. Full-coverage/oracle inference
            # can select thousands of patches on a large CT. Accumulate each
            # small Focus batch immediately into the output and count volumes.
            coord = torch.tensor(coords_all[0], device=input_images.device)
            selected_ids = selected_index.nonzero(as_tuple=False).flatten()
            focus_crops = crops[selected_ids]
            focus_coord = coord[selected_ids]
            focus_context_size = (
                int(getattr(self.args, "focus_context_x", DEFAULT_CROP_SIZE[0])),
                int(getattr(self.args, "focus_context_y", DEFAULT_CROP_SIZE[1])),
                int(getattr(self.args, "focus_context_z", DEFAULT_CROP_SIZE[2])),
            )
            if focus_context_size != DEFAULT_CROP_SIZE:
                focus_crops, expanded_coords = recenter_crops_3d_tensor(
                    input_images[0],
                    focus_coord.detach().cpu().tolist(),
                    crop_size=focus_context_size,
                    source_crop_size=DEFAULT_CROP_SIZE,
                )
                focus_coord = torch.as_tensor(
                    expanded_coords, dtype=torch.long, device=input_images.device
                )
                selected_count = len(expanded_coords)
            infer_batch = getattr(self.args, "focus_batch_size", 4)
            tta_flip_dims = getattr(self.args, "focus_tta_flip_dims", ())
            aggregation_mode = getattr(self.args, "aggregation_mode", "uniform")
            seg_sum = torch.zeros(
                [b, self.args.out_channels, x, y, z], device=input_images.device
            )
            count_map = torch.zeros_like(seg_sum)
            gaussian_map = None
            for i in range(0, selected_count, infer_batch):
                output = focus_probability_with_tta(
                    self.segmentation_model,
                    focus_crops[i:i + infer_batch],
                    flip_dims=tta_flip_dims,
                )
                gaussian_map = accumulate_segmentation_with_coords(
                    seg_sum,
                    count_map,
                    output,
                    focus_coord[i:i + infer_batch],
                    mode=aggregation_mode,
                    gaussian_map=gaussian_map,
                )
            seg_out = finalize_segmentation_aggregation(seg_sum, count_map)

        else:
            compress_ratio = 0.0
            seg_out = torch.zeros([b, self.args.out_channels, x, y, z]).to(input_images.device)

        return seg_out, compress_ratio


def get_dice_reward(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    FOR ABLATION
    Compute Dice Reward per sample using argmax for prediction.
    for ablation studies

    Args:
        pred:   Raw prediction logits (b, 2, h, w, d), without softmax
        label:  Binary ground truth (b, 1, h, w, d), values 0 or 1
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice:   Dice score per sample (b,)
    """
    # Convert label to match pred shape (b, 1, h, w, d) -> (b, h, w, d)
    label_squeezed = label.squeeze(1)  # (b, h, w, d)

    # Get predicted class using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (b, h, w, d)

    # Create binary foreground mask for predictions (1 where predicted as foreground)
    pred_fg = (pred_classes == 1).float()  # (b, h, w, d)

    # Create binary foreground mask for labels
    label_fg = label_squeezed.float()  # (b, h, w, d)

    # Compute intersection and volumes
    intersection = torch.sum(pred_fg * label_fg, dim=(1, 2, 3))  # (b,)
    pred_volume = torch.sum(pred_fg, dim=(1, 2, 3))  # (b,)
    label_volume = torch.sum(label_fg, dim=(1, 2, 3))  # (b,)

    # Compute Dice score
    dice_reward = (2. * intersection + smooth) / (pred_volume + label_volume + smooth)  # (b,)

    return dice_reward


def get_detection_reward(pred: torch.Tensor, label: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute detection Reward per sample using argmax for prediction.

    Args:
        pred:   Raw prediction logits (b, 2, h, w, d), without softmax
        label:  Binary ground truth (b, 1, h, w, d), values 0 or 1
        smooth: Smoothing factor to avoid division by zero

    Returns:
        dice:   Dice score per sample (b,)
    """
    # Convert label to match pred shape (b, 1, h, w, d) -> (b, h, w, d)
    label_squeezed = label.squeeze(1)  # (b, h, w, d)

    # Get predicted class using argmax
    pred_classes = torch.argmax(pred, dim=1)  # (b, h, w, d)

    # Create binary foreground mask for predictions (1 where predicted as foreground)
    pred_fg = (pred_classes == 1).float()  # (b, h, w, d)

    # Create binary foreground mask for labels
    label_fg = label_squeezed.float()  # (b, h, w, d)

    ### Compute intersection and volumes
    intersection = torch.sum(pred_fg * label_fg, dim=(1, 2, 3))  # (b,)
    reward = (intersection > 0).float()

    return reward


def group_relative_advantages(rewards):
    std = rewards.std()
    std = std if std > 1e-2 else 1e-2
    advantages = (rewards - rewards.mean()) / std
    advantages = torch.clamp(advantages, -5, 5)  # 限制在[-5, 5]之间
    return advantages


def masked_group_relative_advantages(rewards, valid):
    """Normalize rewards only across trusted crops; ignored crops stay zero."""
    valid = valid.bool().flatten()
    rewards = rewards.flatten()
    advantages = torch.zeros_like(rewards)
    if not valid.any():
        return advantages
    trusted_rewards = rewards[valid]
    std = trusted_rewards.std(unbiased=False)
    std = std if std > 1e-2 else std.new_tensor(1e-2)
    advantages[valid] = torch.clamp(
        (trusted_rewards - trusted_rewards.mean()) / std,
        -5,
        5,
    )
    return advantages


def get_glance_label(label):
    """
    Optimized processing of label tensor [n, 1, h, w, z]
    Returns tensor where:
        - 0: background (all zeros)
        - 1: contains cancer (any 1s)

    Args:
        label: Input tensor of shape [n, 1, h, w, z]
    Returns:
        Processed tensor of shape [n, h*w*z] with values 0, 1
    """
    # Reshape to [n, h*w*z]
    b, _, x, y, z = label.size()
    label_flat = label.view(b, x * y * z)

    # Create masks for each condition
    has_cancer = (label_flat == 1).any(dim=1, keepdim=True)

    # Combine conditions
    result = torch.zeros_like(label_flat[:, :1])  # Initialize with background
    result[has_cancer] = 1

    return result.long()


def get_glance_loss(actions, label):
    # input actions: (num_crops, 2)
    actions = actions.softmax(1)[:, 1]

    # label: 0 background, 1 cancer
    pos_label = (label == 1).long()
    neg_label = (label != 1).long()

    pos_loss = - (pos_label * torch.log(actions + 1e-6)).sum() / (pos_label.sum() + 1e-6)
    neg_loss = - (neg_label * torch.log(1 - actions + 1e-6)).sum() / (neg_label.sum() + 1e-6)

    loss = pos_loss + neg_loss

    return loss


def gaussian_importance_map(patch_size, device, dtype, sigma_scale=0.125):
    """Return a separable center-emphasizing map for overlap aggregation.

    Focus predictions close to a crop boundary have substantially less 3D
    context than predictions near its center. Uniform averaging treats both as
    equally reliable. This is the same motivation as nnU-Net's Gaussian
    sliding-window weighting. A small positive floor keeps volume boundaries
    numerically stable when they are covered only by a patch edge.
    """
    axes = []
    for size in patch_size:
        coordinate = torch.arange(size, device=device, dtype=torch.float32)
        center = (size - 1) / 2.0
        sigma = max(float(size) * sigma_scale, 1.0)
        axes.append(torch.exp(-0.5 * ((coordinate - center) / sigma).square()))
    importance = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    importance /= importance.max()
    importance.clamp_(min=1e-3)
    return importance.to(dtype=dtype).unsqueeze(0).unsqueeze(0)


def accumulate_segmentation_with_coords(
    zeros, count_map, seg, coord, mode="uniform", gaussian_map=None
):
    """Accumulate one patch batch without retaining earlier Focus outputs.

    Args:
        zeros: [b, c, x, y, z]
        count_map: matching overlap-weight accumulator
        seg: [num_patches, c, patch_x, patch_y, patch_z]
        coord: [num_patches, 3]
    """
    b, c, x, y, z = zeros.size()
    device = zeros.device
    dtype = zeros.dtype

    if mode not in ("uniform", "gaussian"):
        raise ValueError(f"unknown aggregation mode: {mode}")
    if count_map.shape != zeros.shape:
        raise ValueError("count_map and output accumulator must have identical shapes")

    for i in range(seg.shape[0]):
        patch_size = seg[i].shape[1:]  # [patch_x, patch_y, patch_z]

        if len(patch_size) != 3:
            print(f"Warning: patch_size has {len(patch_size)} dimensions, expected 3. patch_size: {patch_size}")
            continue

        if coord[i].shape[0] == 3:
            x_start, y_start, z_start = coord[i]
            x_end = x_start + patch_size[0]
            y_end = y_start + patch_size[1]
            z_end = z_start + patch_size[2]
        elif coord[i].shape[0] == 6:
            x_start, y_start, z_start, x_end, y_end, z_end = coord[i]
        else:
            print(f"Warning: coord[i] has {coord[i].shape[0]} dimensions, expected 3 or 6. coord[i]: {coord[i]}")
            continue

        x_start = max(0, int(x_start))
        y_start = max(0, int(y_start))
        z_start = max(0, int(z_start))
        x_end = min(x, int(x_end))
        y_end = min(y, int(y_end))
        z_end = min(z, int(z_end))

        if x_start >= x_end or y_start >= y_end or z_start >= z_end:
            print(
                f"Warning: Invalid coordinates for patch {i}: x({x_start}:{x_end}), y({y_start}:{y_end}), z({z_start}:{z_end})")
            continue

        actual_patch_size = [x_end - x_start, y_end - y_start, z_end - z_start]
        if actual_patch_size != list(patch_size):
            seg_patch = F.interpolate(seg[i:i + 1], size=actual_patch_size, mode='trilinear', align_corners=False)
        else:
            seg_patch = seg[i:i + 1]

        if mode == "gaussian":
            if gaussian_map is None or tuple(gaussian_map.shape[2:]) != tuple(patch_size):
                gaussian_map = gaussian_importance_map(patch_size, device, dtype)
            if actual_patch_size != list(patch_size):
                weight_map = F.interpolate(
                    gaussian_map,
                    size=actual_patch_size,
                    mode="trilinear",
                    align_corners=False,
                )
            else:
                weight_map = gaussian_map
        else:
            weight_map = torch.ones([1, 1] + actual_patch_size, dtype=dtype, device=device)

        zeros[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += seg_patch * weight_map
        count_map[:, :, x_start:x_end, y_start:y_end, z_start:z_end] += weight_map

    return gaussian_map


def finalize_segmentation_aggregation(zeros, count_map):
    """Normalize an accumulated segmentation, leaving uncovered voxels zero."""
    safe_count = count_map.clamp_min(1)
    return zeros / safe_count


def aggregate_segmentation_with_coords(zeros, seg, coord, mode="uniform"):
    """Aggregate a complete patch tensor; retained for legacy callers/tests."""
    count_map = torch.zeros_like(zeros)
    accumulate_segmentation_with_coords(
        zeros, count_map, seg, coord, mode=mode
    )
    return finalize_segmentation_aggregation(zeros, count_map)

