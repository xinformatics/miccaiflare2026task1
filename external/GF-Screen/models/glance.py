from models.resnet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


DEFAULT_CROP_SIZE = (96, 96, 64)


def random_crop_3d_tensor(tensor, crop_size=DEFAULT_CROP_SIZE, num_crops=4):
    if tensor.dim() == 3:
        D, H, W = tensor.shape
        has_channel = False
    elif tensor.dim() == 4:  # 如果有 channel 维度 (C, D, H, W)
        C, D, H, W = tensor.shape
        has_channel = True
    else:
        raise ValueError("Input tensor must be 3D (D,H,W) or 4D (C,D,H,W)")

    c_d, c_h, c_w = crop_size

    if c_d > D or c_h > H or c_w > W:
        raise ValueError(f"Crop size {crop_size} is larger than input tensor size {(D, H, W)}")

    crops = []
    coords = []

    for _ in range(num_crops):
        # 随机生成起始坐标
        start_d = random.randint(0, D - c_d)
        start_h = random.randint(0, H - c_h)
        start_w = random.randint(0, W - c_w)

        end_d = start_d + c_d
        end_h = start_h + c_h
        end_w = start_w + c_w

        if has_channel:
            crop = tensor[:, start_d:end_d, start_h:end_h, start_w:end_w]
        else:
            crop = tensor[start_d:end_d, start_h:end_h, start_w:end_w]

        crops.append(crop.unsqueeze(0))
        coords.append((start_d, start_h, start_w))

    return crops, coords


def _axis_crop_starts(size, crop, overlap_ratio, grid_mode):
    overlap = int(crop * overlap_ratio)
    stride = crop - overlap
    count = (size - overlap + stride - 1) // stride
    if count <= 1:
        return [0]
    if grid_mode == "legacy":
        return [min(index * stride, size - crop) for index in range(count)]
    if grid_mode == "even":
        # Match the legacy crop count but distribute the residual distance
        # across every interval. This avoids a nearly duplicated final crop
        # when (size - crop) is not divisible by the nominal stride.
        extent = size - crop
        return [round(index * extent / (count - 1)) for index in range(count)]
    raise ValueError(f"unknown grid mode: {grid_mode}")


def crop_3d_tensor(tensor, crop_size=DEFAULT_CROP_SIZE, overlap_ratio=0.75, grid_mode="legacy"):
    if tensor.dim() == 3:
        D, H, W = tensor.shape
    elif tensor.dim() == 4:
        C, D, H, W = tensor.shape
    else:
        raise ValueError("Input tensor must be 3D (D,H,W) or 4D (C,D,H,W)")

    c_d, c_h, c_w = crop_size

    starts_d = _axis_crop_starts(D, c_d, overlap_ratio, grid_mode)
    starts_h = _axis_crop_starts(H, c_h, overlap_ratio, grid_mode)
    starts_w = _axis_crop_starts(W, c_w, overlap_ratio, grid_mode)

    crops = []
    coords = []
    for start_d in starts_d:
        for start_h in starts_h:
            for start_w in starts_w:
                end_d = start_d + c_d
                end_h = start_h + c_h
                end_w = start_w + c_w

                if tensor.dim() == 3:
                    crop = tensor[start_d:end_d, start_h:end_h, start_w:end_w]
                else:  # dim=4, (C,D,H,W)
                    crop = tensor[:, start_d:end_d, start_h:end_h, start_w:end_w]

                crops.append(crop.unsqueeze(0))

                coord = (start_d, start_h, start_w)
                coords.append(coord)

    return crops, coords


def recenter_crops_3d_tensor(
    tensor,
    coords,
    crop_size,
    source_crop_size=DEFAULT_CROP_SIZE,
):
    """Expand crop coordinates around their original physical centres.

    Glance remains calibrated on the released 96x96x64 grid.  This helper
    lets Focus receive a larger field of view around the same selected centres
    without asking Glance to classify out-of-distribution crop sizes. Repeated
    expanded starts are deduplicated in stable order.
    """
    if tensor.dim() == 3:
        spatial_shape = tuple(int(value) for value in tensor.shape)
    elif tensor.dim() == 4:
        spatial_shape = tuple(int(value) for value in tensor.shape[1:])
    else:
        raise ValueError("Input tensor must be 3D (D,H,W) or 4D (C,D,H,W)")
    target = tuple(int(value) for value in crop_size)
    source = tuple(int(value) for value in source_crop_size)
    if len(target) != 3 or len(source) != 3 or min(*target, *source) < 1:
        raise ValueError("crop sizes must contain three positive integers")
    if any(size > extent for size, extent in zip(target, spatial_shape)):
        raise ValueError(
            f"Crop size {target} is larger than input tensor size {spatial_shape}"
        )

    expanded_coords = []
    seen = set()
    for coordinate in coords:
        if len(coordinate) != 3:
            raise ValueError("coords must contain (D,H,W) crop starts")
        center = [int(start) + size // 2 for start, size in zip(coordinate, source)]
        expanded = tuple(
            max(0, min(axis_center - size // 2, extent - size))
            for axis_center, size, extent in zip(center, target, spatial_shape)
        )
        if expanded not in seen:
            seen.add(expanded)
            expanded_coords.append(expanded)

    crops = []
    for start in expanded_coords:
        slices = tuple(slice(value, value + size) for value, size in zip(start, target))
        crop = tensor[slices] if tensor.dim() == 3 else tensor[(slice(None), *slices)]
        crops.append(crop.unsqueeze(0))
    if not crops:
        leading = () if tensor.dim() == 3 else (tensor.shape[0],)
        return tensor.new_empty((0, *leading, *target)), expanded_coords
    return torch.cat(crops, dim=0), expanded_coords


def crop_labels_with_coords(labels, coords, crop_size=DEFAULT_CROP_SIZE,):
    c_d, c_h, c_w = crop_size
    crop_labels = []

    for coord in coords:

        start_d, start_h, start_w = coord
        end_d = start_d + c_d
        end_h = start_h + c_h
        end_w = start_w + c_w

        if labels.dim() == 3:
            crop = labels[start_d:end_d, start_h:end_h, start_w:end_w]
        else:  # dim=4 (C,D,H,W)
            crop = labels[:, start_d:end_d, start_h:end_h, start_w:end_w]

        crop_labels.append(crop.unsqueeze(0))

    crop_labels = torch.concat(crop_labels, dim=0)

    return crop_labels


def select_crops_from_proposal(proposal, coords, crop_size=DEFAULT_CROP_SIZE, min_voxels=1):
    """Return crop-selection flags for a binary external proposal mask.

    This deliberately uses the exact inference grid and crop implementation,
    so proposal-guided Focus inference cannot introduce a coordinate mismatch.
    ``proposal`` is one channel in the already transformed image space.
    """
    if min_voxels < 1:
        raise ValueError("proposal min_voxels must be at least 1")
    maximum_count = proposal.numel()
    count_dtype = torch.int32
    if maximum_count > torch.iinfo(torch.int32).max:
        count_dtype = torch.int64

    if proposal.dim() == 4:
        # Preserve the legacy behavior for a possible multi-channel proposal:
        # every positive channel voxel contributes to the crop count.
        foreground = proposal.gt(0).sum(dim=0, dtype=count_dtype)
    elif proposal.dim() == 3:
        foreground = proposal.gt(0).to(dtype=count_dtype)
    else:
        raise ValueError("proposal must be 3D (D,H,W) or 4D (C,D,H,W)")

    # The former implementation concatenated one full proposal crop for every
    # inference-grid coordinate. On a large CT that allocation scales as
    # O(number_of_crops * crop_volume) and can consume several GiB before Focus
    # batching begins. A summed-volume table computes exactly the same counts
    # with one O(volume) allocation and O(number_of_crops) indexed lookups.
    integral = foreground
    for dimension in range(3):
        integral = integral.cumsum(dimension, dtype=count_dtype)
    integral = F.pad(integral, (1, 0, 1, 0, 1, 0))

    coordinates = torch.as_tensor(coords, dtype=torch.long, device=proposal.device)
    if coordinates.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=proposal.device)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coords must contain (D,H,W) crop starts")
    crop = torch.as_tensor(crop_size, dtype=torch.long, device=proposal.device)
    starts = coordinates
    ends = starts + crop
    spatial_shape = torch.as_tensor(
        foreground.shape, dtype=torch.long, device=proposal.device
    )
    if bool((starts < 0).any()) or bool((ends > spatial_shape).any()):
        raise ValueError("proposal crop coordinates fall outside the proposal volume")

    d0, h0, w0 = starts.unbind(1)
    d1, h1, w1 = ends.unbind(1)
    counts = (
        integral[d1, h1, w1]
        - integral[d0, h1, w1]
        - integral[d1, h0, w1]
        - integral[d1, h1, w0]
        + integral[d0, h0, w1]
        + integral[d0, h1, w0]
        + integral[d1, h0, w0]
        - integral[d0, h0, w0]
    )
    return counts.ge(int(min_voxels))


class Glance_model(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(
        self,
        backbone='resnet18',
        grid_mode='legacy',
        box_localization=False,
        center_localization=False,
    ):
        super(Glance_model, self).__init__()
        self.grid_mode = grid_mode
        self.box_localization = bool(box_localization)
        self.center_localization = bool(center_localization)
        if backbone == 'resnet10':
            self.backbone = resnet10(pretrained=False, n_input_channels=1, num_classes=2, feed_forward=False,
                                     shortcut_type='B', bias_downsample=False)

        else:
            self.backbone = resnet18(pretrained=False, n_input_channels=1, num_classes=2, feed_forward=False,
                                     shortcut_type='A', bias_downsample=True)

        self.layer = nn.Sequential(
            nn.Linear(512, 2),
        )
        # Optional training-time auxiliary head. Coordinates are normalized
        # (start_d,start_h,start_w,end_d,end_h,end_w) and constrained to [0,1]
        # only by the loss. The deployed crop gate continues to use ``layer``;
        # no ground-truth or predicted box is required during inference.
        if self.box_localization:
            self.box_layer = nn.Linear(512, 6)

        # Optional training-only dense lesion-centre head.  The layer-2
        # feature map is at quarter input resolution (24x24x16 for the
        # standard crop), preserving small-lesion evidence that can disappear
        # before global pooling.  Deployment still uses ``layer`` so legacy
        # inference and checkpoint behavior are unchanged after this head is
        # stripped.
        if self.center_localization:
            self.center_layer = nn.Conv3d(128, 1, kernel_size=1)

    def get_actions(self, x, return_boxes=False, return_centers=False):
        # inference with batch, to avoid large batch --> out of memory
        infer_batch = 16  # max batch
        total_samples = x.size()[0]
        results = []
        box_results = []
        center_results = []
        for i in range(0, total_samples, infer_batch):
            batch = x[i:i + infer_batch]
            if return_centers:
                if not self.center_localization:
                    raise RuntimeError(
                        "center predictions requested without center-localization head"
                    )
                features, layer2 = self.backbone.forward_with_layer2(batch)
                center_results.append(self.center_layer(layer2))
            else:
                features = self.backbone(batch)
            results.append(self.layer(features))
            if return_boxes:
                if not self.box_localization:
                    raise RuntimeError(
                        "box predictions requested without box-localization head"
                    )
                box_results.append(self.box_layer(features))

        # Concatenate all results
        actions = torch.cat(results, dim=0)
        auxiliary = []
        if return_boxes:
            auxiliary.append(torch.cat(box_results, dim=0))
        if return_centers:
            auxiliary.append(torch.cat(center_results, dim=0))
        if auxiliary:
            return (actions, *auxiliary)
        return actions

    def forward(self, input_images, return_center_predictions=False):
        # for validation

        b = input_images.size()[0]
        xs = []
        coords_all = []

        for i in range(b):
            input_image = input_images[i][0]
            crops, coords = crop_3d_tensor(input_image, grid_mode=self.grid_mode)
            xs += crops
            coords_all.append(coords)
        # coords_all length: batch_size. each length: num of crops

        crops = torch.concat(xs, dim=0).unsqueeze(1)
        action_output = self.get_actions(
            crops, return_centers=return_center_predictions
        )
        if return_center_predictions:
            actions, centers = action_output
            return crops, coords_all, actions, centers
        return crops, coords_all, action_output

    def forward_with_labels(
        self,
        input_images,
        input_labels,
        training=True,
        return_box_predictions=False,
        return_center_predictions=False,
    ):
        # for training with labels
        b = input_images.size()[0]
        xs = []
        coords_all = []

        for i in range(b):
            input_image = input_images[i][0]

            if training:
                crops, coords = random_crop_3d_tensor(input_image, num_crops=1)
            else:
                crops, coords = crop_3d_tensor(input_image, grid_mode=self.grid_mode)

            xs += crops
            coords_all.append(coords)
        # coords_all length: batch_size. each length: num of crops

        crops = torch.concat(xs, dim=0).unsqueeze(1)
        action_output = self.get_actions(
            crops,
            return_boxes=return_box_predictions,
            return_centers=return_center_predictions,
        )
        if not (return_box_predictions or return_center_predictions):
            actions = action_output
            auxiliary = ()
        else:
            actions, *auxiliary = action_output

        labels_all = []
        # crop labels:
        for i in range(b):
            label = input_labels[i]
            coords_each = coords_all[i]
            crop_labels = crop_labels_with_coords(label, coords_each)
            labels_all.append(crop_labels)
        output_labels = torch.concat(labels_all, dim=0)

        output = (crops, coords_all, output_labels, actions)
        return output + tuple(auxiliary)
