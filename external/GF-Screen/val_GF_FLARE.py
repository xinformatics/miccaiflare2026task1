
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

import argparse
from collections import Counter
from pathlib import Path
from torch.cuda.amp import autocast
# from monai.data import decollate_batch
from monai.transforms import *
from utils.utils import *
from monai import transforms
from monai.data import *
import resource
from models.GF_Screen_RL import GFScreen_RL
from inference_naming import incomplete_inputs, prediction_name
import time

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument(
    "--test_data_path", default="./data/FLARE-Task1-Pancancer/validation/Validation-Hidden-Images/", type=str, help="test_data_path")
parser.add_argument(
    "--save_prediction_path", default="./pred/FLARE25_Val_GF_full_abdomen/", type=str, help="test_prediction_path")
parser.add_argument(
    "--trained_pth", default="./runs/logs/model_final.pt", type=str, help="trained checkpoint directory")
parser.add_argument("--glance_backbone", default='resnet18', help="pre-trained root")
parser.add_argument(
    "--glance_box_localization",
    action="store_true",
    help="instantiate the auxiliary Glance box head stored in compatible checkpoints",
)
parser.add_argument(
    "--glance_center_localization",
    action="store_true",
    help="instantiate the optional dense Glance centre head stored in compatible checkpoints",
)
parser.add_argument(
    "--glance_center_threshold",
    default=None,
    type=float,
    help=(
        "optional sigmoid peak threshold for routing a crop through Focus via "
        "the dense Glance centre head; requires --glance_center_localization"
    ),
)

parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument(
    "--focus_context_x", default=96, type=int,
    help="Focus inference crop x size; Glance remains on its released 96-voxel grid",
)
parser.add_argument(
    "--focus_context_y", default=96, type=int,
    help="Focus inference crop y size; Glance remains on its released 96-voxel grid",
)
parser.add_argument(
    "--focus_context_z", default=64, type=int,
    help="Focus inference crop z size; Glance remains on its released 64-slice grid",
)
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument(
    "--recursive_inputs",
    action="store_true",
    help=(
        "discover NIfTI inputs recursively below --test_data_path; basenames "
        "must be unique because predictions are written to one flat directory"
    ),
)
parser.add_argument(
    "--case_names",
    default=None,
    type=str,
    help=(
        "optional comma-separated case basenames for targeted inference; accepts "
        "either FLARE label names (case.nii.gz) or image names (case_0000.nii.gz)"
    ),
)
parser.add_argument("--num_shards", default=1, type=int, help="number of deterministic input shards")
parser.add_argument("--shard_id", default=0, type=int, help="zero-based shard assigned to this process")
parser.add_argument("--glance_threshold", default=0.9, type=float, help="Glance lesion-crop probability threshold")
parser.add_argument("--lesion_threshold", default=0.5, type=float, help="Focus foreground probability threshold")
parser.add_argument(
    "--lesion_thresholds",
    default=None,
    type=str,
    help="optional comma-separated Focus thresholds; writes one subdirectory per threshold",
)
parser.add_argument("--focus_batch_size", default=4, type=int, help="Focus crop inference micro-batch")
parser.add_argument(
    "--focus_tta_axes",
    default="",
    type=str,
    help=(
        "optional unique crop-space flip axes chosen from x,y,z; for example "
        "x averages original/LR-flipped Focus probabilities and xy uses three views"
    ),
)
parser.add_argument(
    "--aggregation_mode",
    default="uniform",
    choices=("uniform", "gaussian"),
    help="overlap aggregation for Focus probabilities",
)
parser.add_argument(
    "--grid_mode",
    default="legacy",
    choices=("legacy", "even"),
    help="placement of the fixed-count 96x96x64 inference crop grid",
)
parser.add_argument("--skip_existing", action="store_true", help="skip cases written for every requested threshold")
parser.add_argument(
    "--proposal_mask_path",
    default=None,
    type=str,
    help=(
        "optional directory of full-resolution binary masks; every crop "
        "overlapping a proposal is sent to Focus in addition to Glance-selected crops"
    ),
)
parser.add_argument(
    "--proposal_min_voxels",
    default=1,
    type=int,
    help="minimum proposal voxels required inside an inference crop",
)
parser.add_argument(
    "--proposal_only",
    action="store_true",
    help="select Focus crops only from --proposal_mask_path, ignoring Glance",
)
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


def get_test_loader(args):
    """
    Creates training transforms, constructs a dataset, and returns a dataloader.

    Args:
        args: Command line arguments containing dataset paths and hyperparameters.
    """
    spatial_keys = ["image", "proposal"] if args.proposal_mask_path else ["image"]
    spacing_modes = ("bilinear", "nearest") if args.proposal_mask_path else ("bilinear",)
    test_transforms = transforms.Compose([
        LoadImaged(keys=spatial_keys),
        EnsureChannelFirstd(keys=spatial_keys),
        Orientationd(keys=spatial_keys, axcodes="RAS"),
        Spacingd(keys=spatial_keys, pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=spacing_modes),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=spatial_keys, source_key="image"),
        SpatialPadd(keys=spatial_keys, spatial_size=(
                    max(args.roi_x, args.focus_context_x),
                    max(args.roi_y, args.focus_context_y),
                    max(args.roi_z, args.focus_context_z),
                    ),
                    mode='constant'),
    ])

    # constructing training dataset
    test_img = []
    test_name = []
    test_proposal = []

    if args.recursive_inputs:
        discovered = sorted(
            path
            for path in Path(args.test_data_path).rglob("*")
            if path.is_file() and (path.name.endswith(".nii.gz") or path.name.endswith(".nii"))
        )
        basenames = [path.name for path in discovered]
        duplicate_basenames = sorted(
            name for name, count in Counter(basenames).items() if count > 1
        )
        if duplicate_basenames:
            raise ValueError(
                "recursive inputs contain duplicate basenames: "
                f"{duplicate_basenames[:10]}"
            )
        dataset_list = discovered
    else:
        dataset_list = [Path(args.test_data_path) / name for name in os.listdir(args.test_data_path)]
        dataset_list = sorted(path for path in dataset_list if path.is_file())
    if args.case_names:
        requested = {
            name.strip().removesuffix("_0000.nii.gz").removesuffix(".nii.gz")
            for name in args.case_names.split(",")
            if name.strip()
        }
        if not requested:
            raise ValueError("--case_names did not contain any case basenames")
        discovered_by_case = {
            path.name.removesuffix("_0000.nii.gz").removesuffix(".nii.gz"): path
            for path in dataset_list
        }
        missing = sorted(requested - discovered_by_case.keys())
        if missing:
            raise FileNotFoundError(
                f"--case_names requested {len(missing)} absent cases; first={missing[0]}"
            )
        dataset_list = sorted(discovered_by_case[name] for name in requested)
        print(f"targeted inference: retained {len(dataset_list)} requested cases", flush=True)
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"invalid shard {args.shard_id}/{args.num_shards}")
    dataset_list = dataset_list[args.shard_id::args.num_shards]
    if args.skip_existing:
        thresholds = (
            [float(value) for value in args.lesion_thresholds.split(",")]
            if args.lesion_thresholds
            else [float(args.lesion_threshold)]
        )
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError(f"invalid lesion thresholds: {thresholds}")
        directories = [
            Path(args.save_prediction_path)
            / (f"lesion_{threshold:.2f}" if args.lesion_thresholds else "")
            for threshold in thresholds
        ]
        original_count = len(dataset_list)
        dataset_list = incomplete_inputs(dataset_list, directories)
        print(
            f"early skip-existing: retained {len(dataset_list)}/{original_count} "
            "shard inputs",
            flush=True,
        )

    for item in dataset_list:
        name = item.name
        print(name)
        test_img_path = str(item)

        test_img.append(test_img_path)
        test_name.append(name)
        if args.proposal_mask_path:
            # FLARE-style inputs use ``case_0000.nii.gz``, while some of the
            # lesion-free safety cases are already named ``case.nii.gz``.
            # Stripping only the FLARE suffix made the latter resolve to
            # ``case.nii.gz.nii.gz`` and aborted guided inference.
            if name.endswith("_0000.nii.gz"):
                proposal_name = name.removesuffix("_0000.nii.gz") + ".nii.gz"
            else:
                proposal_name = name.removesuffix(".nii.gz") + ".nii.gz"
            proposal_path = os.path.join(args.proposal_mask_path, proposal_name)
            if not os.path.isfile(proposal_path):
                raise FileNotFoundError(f"missing proposal mask: {proposal_path}")
            test_proposal.append(proposal_path)

    if args.proposal_mask_path:
        data_dicts_test = [
            {'image': image, 'proposal': proposal, 'name': name}
            for image, proposal, name in zip(test_img, test_proposal, test_name)
        ]
    else:
        data_dicts_test = [{'image': image, 'name': name}
                           for image, name in zip(test_img, test_name)]

    print('test len {}'.format(len(data_dicts_test)))

    test_ds = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True
    )
    return test_loader, test_transforms


def main():
    args = parser.parse_args()
    focus_context = (args.focus_context_x, args.focus_context_y, args.focus_context_z)
    released_context = (args.roi_x, args.roi_y, args.roi_z)
    if any(size < released for size, released in zip(focus_context, released_context)):
        parser.error("Focus context dimensions cannot be smaller than the released ROI")
    if any(size % 32 for size in focus_context):
        parser.error("Focus context dimensions must be divisible by 32 for SwinUNETR")
    if args.glance_center_threshold is not None:
        if not args.glance_center_localization:
            parser.error(
                "--glance_center_threshold requires --glance_center_localization"
            )
        if not 0.0 < args.glance_center_threshold < 1.0:
            parser.error("--glance_center_threshold must be strictly between 0 and 1")
    if args.proposal_only and not args.proposal_mask_path:
        raise ValueError("--proposal_only requires --proposal_mask_path")
    if args.proposal_min_voxels < 1:
        raise ValueError("--proposal_min_voxels must be at least 1")
    normalized_tta_axes = args.focus_tta_axes.lower().replace(",", "").replace(" ", "")
    if any(axis not in "xyz" for axis in normalized_tta_axes):
        raise ValueError("--focus_tta_axes must contain only x, y, and z")
    if len(set(normalized_tta_axes)) != len(normalized_tta_axes):
        raise ValueError("--focus_tta_axes cannot repeat an axis")
    axis_to_dimension = {"x": 2, "y": 3, "z": 4}
    args.focus_tta_flip_dims = tuple(axis_to_dimension[axis] for axis in normalized_tta_axes)
    test_loader, test_transforms = get_test_loader(args)

    lesion_thresholds = (
        [float(value) for value in args.lesion_thresholds.split(",")]
        if args.lesion_thresholds
        else [args.lesion_threshold]
    )
    if any(not 0.0 < value < 1.0 for value in lesion_thresholds):
        raise ValueError(f"invalid lesion thresholds: {lesion_thresholds}")

    model = GFScreen_RL(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # These checkpoints are produced locally by GF_RL_train and include
    # optimizer/scheduler metadata containing NumPy scalars. PyTorch 2.6's
    # weights_only=True default rejects that trusted training metadata.
    model_dict = torch.load(args.trained_pth, weights_only=False)["state_dict"]
    model.load_state_dict(model_dict, strict=True)

    model.eval()
    model.to(device)

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    prediction_paths = {}
    post_transforms = {}
    for threshold in lesion_thresholds:
        output_path = args.save_prediction_path
        if args.lesion_thresholds:
            output_path = os.path.join(output_path, f"lesion_{threshold:.2f}")
        os.makedirs(output_path, exist_ok=True)
        prediction_paths[threshold] = output_path
        post_transforms[threshold] = Compose([
            EnsureTyped(keys=["pred"]),
            Invertd(keys=["pred"],
                    transform=test_transforms,
                    orig_keys="image",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                    nearest_interp=True,
                    to_tensor=True),
            AsDiscreted(keys="pred", argmax=False, to_onehot=None),
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_path,
                       separate_folder=False, folder_layout=None,
                       resample=False),
        ])

    total_compress_ratio = 0.0
    count = 0

    overall_duration = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            torch.cuda.empty_cache()
            start_time = time.time()

            data = batch_data["image"]
            data = data.cuda()
            proposal = batch_data.get("proposal")
            if proposal is not None:
                proposal = proposal.cuda()
            name = batch_data['name'][0]
            print(name)

            final_name = prediction_name(name)
            if args.skip_existing and all(
                os.path.isfile(os.path.join(prediction_paths[threshold], final_name))
                for threshold in lesion_thresholds
            ):
                print("already complete; skipping")
                continue

            with autocast(enabled=True):
                logits, compress_ratio = model.valid(data, proposal_mask=proposal)

            total_compress_ratio += compress_ratio
            count += 1
            mean_compress_ratio = total_compress_ratio / count
            print('compress_ratio:', mean_compress_ratio)

            duration = time.time() - start_time
            overall_duration += duration
            average_duration = overall_duration / count

            print('Duration:', duration)
            print('Average Duration:', average_duration)

            print('\n')

            for threshold in lesion_thresholds:
                output = (logits[:, 1] >= threshold).long()
                threshold_batch = dict(batch_data)
                threshold_batch['pred'] = output.unsqueeze(1)
                _ = [post_transforms[threshold](item) for item in
                     decollate_batch(threshold_batch)]

                output_path = prediction_paths[threshold]
                os.rename(os.path.join(output_path, name[:-7] + '_trans.nii.gz'),
                          os.path.join(output_path, final_name))


if __name__ == "__main__":
    main()
    args = parser.parse_args()
