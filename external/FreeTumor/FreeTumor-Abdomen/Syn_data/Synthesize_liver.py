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
import os
from functools import partial
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
# from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import *
from monai.utils.enums import MetricReduction
from monai.handlers import StatsHandler, from_engine
import matplotlib.pyplot as plt
from PIL import Image
from monai import data, transforms
from monai.data import *
from models.TumorGAN import *

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument(
    "--test_data_path", default="/data/FreeTumor/xxx/imagesTr/", type=str, help="test_data_path")
parser.add_argument(
    "--test_label_path", default="/data/FreeTumor/xxx/labelsTr/", type=str, help="test_label_path")
parser.add_argument(
    "--save_img_path", default="./syn_data_save/liver/img", type=str, help="test_img_path")
parser.add_argument(
    "--save_lab_path", default="./syn_data_save/liver/lab", type=str, help="test_lab_path")
parser.add_argument(
    "--TGAN_checkpoint", default="./runs/logs_syn/model_final.pt", type=str, help="trained checkpoint directory")
parser.add_argument(
    "--pretrained_dir", default="./baseline/model_baseline_segmentor.pt", type=str, help="baseline segmentor"
)
roi = 96
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=7, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")


def get_test_loader(args):
    """
    Creates training transforms, constructs a dataset, and returns a dataloader.

    Args:
        args: Command line arguments containing dataset paths and hyperparameters.
    """
    test_transforms = transforms.Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),
    ])

    # constructing training dataset
    test_img = []
    test_label = []
    test_name = []

    dataset_list = os.listdir(args.test_data_path)

    for item in dataset_list:
        name = item
        print(name)
        test_img_path = os.path.join(args.test_data_path, name)
        test_label_path = os.path.join(args.test_label_path, name[:-12]+'.nii.gz')

        test_img.append(test_img_path)
        test_label.append(test_label_path)
        test_name.append(name)

    data_dicts_test = [{'image': image, "label": label, 'name': name}
                        for image, label, name in zip(test_img, test_label, test_name)]

    print('test len {}'.format(len(data_dicts_test)))

    test_ds = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True
    )
    return test_loader, test_transforms


def main():
    args = parser.parse_args()

    test_loader, test_transforms = get_test_loader(args)

    Tgan = TGAN(args).cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.TGAN_checkpoint, map_location=torch.device('cpu'))
    state_dict = model_dict["state_dict"]
    Tgan.load_state_dict(state_dict, strict=True)
    Tgan.eval()
    Tgan.to(device)
    print('Load trained TumorGAN')

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    post_img_transforms = Compose([EnsureTyped(keys=["new_img"]),
                               Invertd(keys=["new_img"],
                                       transform=test_transforms,
                                       orig_keys="image",
                                       meta_keys="new_img_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               SaveImaged(keys="new_img", meta_keys="new_img_meta_dict", output_dir=args.save_img_path,
                                          separate_folder=False, folder_layout=None,
                                          resample=False),
                               ])
    post_lab_transforms = Compose([EnsureTyped(keys=["new_lab"]),
                               Invertd(keys=["new_lab"],
                                       transform=test_transforms,
                                       orig_keys="image",
                                       meta_keys="new_lab_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               # AsDiscreted(keys="new_lab", argmax=False, to_onehot=None),
                               SaveImaged(keys="new_lab", meta_keys="new_lab_meta_dict", output_dir=args.save_lab_path,
                                          separate_folder=False, folder_layout=None,
                                          resample=False),
                               ])

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            torch.cuda.empty_cache()

            img = batch_data["image"]
            img = img.cuda()

            lab = batch_data["label"]
            lab = lab.cuda()

            name = batch_data['name'][0]

            try_times = 0
            output_flag = False

            with autocast(enabled=True):
                while try_times < 10 and output_flag is False:
                    # try 10 times
                    try_times += 1
                    new_img, new_lab = syn_data(img, lab, Tgan, args)
                    print(try_times, torch.unique(lab), torch.unique(new_lab))
                    if 2 in list(torch.unique(new_lab.long())):
                        output_flag = True

            if output_flag is True:
                batch_data['new_img'] = new_img
                for i in decollate_batch(batch_data):
                    post_img_transforms(i)

                os.rename(os.path.join(args.save_img_path, name[:-7] + '_trans.nii.gz'),
                          os.path.join(args.save_img_path, name[:-7] + '_new_img.nii.gz'))

                batch_data['new_lab'] = new_lab
                for i in decollate_batch(batch_data):
                    post_lab_transforms(i)

                os.rename(os.path.join(args.save_lab_path, name[:-7] + '_trans.nii.gz'),
                          os.path.join(args.save_lab_path, name[:-7] + '_new_lab.nii.gz'))


def get_3D_position(label):
    lab_numpy = label[0][0].data.cpu().numpy()

    x_start, x_end = np.where(np.any(lab_numpy, axis=(1, 2)))[0][[0, -1]]
    y_start, y_end = np.where(np.any(lab_numpy, axis=(0, 2)))[0][[0, -1]]
    z_start, z_end = np.where(np.any(lab_numpy, axis=(0, 1)))[0][[0, -1]]

    # Padding to ensure at least 96 units distance
    pad_value = 96

    x_diff = x_end - x_start
    y_diff = y_end - y_start
    z_diff = z_end - z_start

    if x_diff < pad_value:
        pad_x = (pad_value - x_diff) // 2
        x_start = max(0, x_start - pad_x)
        x_end = min(lab_numpy.shape[0], x_end + pad_x)

    if y_diff < pad_value:
        pad_y = (pad_value - y_diff) // 2
        y_start = max(0, y_start - pad_y)
        y_end = min(lab_numpy.shape[1], y_end + pad_y)

    if z_diff < pad_value:
        pad_z = (pad_value - z_diff) // 2
        z_start = max(0, z_start - pad_z)
        z_end = min(lab_numpy.shape[2], z_end + pad_z)

    print(x_end - x_start, y_end - y_start, z_end - z_start)
    return x_start, x_end, y_start, y_end, z_start, z_end


def syn_data(img, lab, Tgan, args):
    x_start, x_end, y_start, y_end, z_start, z_end = get_3D_position(lab)

    cut_img = img[:, :, x_start:x_end, y_start:y_end, z_start:z_end]
    cut_mask = lab[:, :, x_start:x_end, y_start:y_end, z_start:z_end]

    mode = 'generate'
    _, _, _, _, syn_img, syn_lab = Tgan(cut_img, cut_mask, mode, args)

    new_img, new_lab = img.clone(), lab.clone()
    new_img[:, :, x_start:x_end, y_start:y_end, z_start:z_end] = syn_img
    new_lab[:, :, x_start:x_end, y_start:y_end, z_start:z_end] = syn_lab

    return new_img, new_lab.long()


if __name__ == "__main__":
    main()