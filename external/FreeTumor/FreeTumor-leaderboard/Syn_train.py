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
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from Syn_trainer import run_training
from utils import get_loader
from models.TumorGAN import *
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--baseline_seg_dir", default="./baseline/model_baseline_segmentor.pt", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--use_ssl_pretrained", default=True, help="use self-supervised pretrained weights")
parser.add_argument("--data", default="syn", type=str, help="data name")
parser.add_argument("--out_channels", default=7, type=int, help="number of output channels")
parser.add_argument(
    "--pretrained_model_name",
    default="model_bestVal.pt",
    type=str,
    help="pretrained model name",
)

parser.add_argument("--resume_ckpt", default=True, help="resume training from pretrained checkpoint")
parser.add_argument(
    "--TGAN_checkpoint",
    default="./runs/logs_syn/model_final.pt",
    type=str,
    help="pretrained model name",
)

parser.add_argument("--pos", default=1, type=int, help="number of positive sample")
parser.add_argument("--neg", default=0, type=int, help="number of negative sample")

roi = 96
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")

parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", default=True, help="do NOT use amp for training")

parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")

parser.add_argument("--local-rank", type=int, default=0, help="local rank")
parser.add_argument("--dist-url", default="env://", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
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
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")

parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    main_worker(args=args)


def main_worker(args):
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0

    if args.distributed:
        args.device = "cuda:%d" % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if args.rank == 0:
            print(
                "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
                % (args.rank, args.world_size)
            )
    else:
        torch.cuda.set_device(0)
        print("Training with a single process on 1 GPUs.")
    assert args.rank >= 0

    loader = get_loader(args)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = TGAN(args)
    model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0 or args.distributed is False:
        print("Total parameters count", pytorch_total_params)

    if args.resume_ckpt:
        # should be before distributed
        if args.rank == 0 or args.distributed is False:
            print('resume from previous checkpoints')
        model_dict = torch.load(args.TGAN_checkpoint, map_location=torch.device('cpu'))
        state_dict = model_dict["state_dict"]
        model.load_state_dict(state_dict, strict=True)
        print("Resume TGAN weights !!!")

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=args.optim_lr, betas=(0.0, 0.999))
        optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=args.optim_lr, betas=(0.0, 0.999))
    else:
        optimizerG = torch.optim.Adam(model.netG.parameters(), lr=args.optim_lr, betas=(0.0, 0.999))
        optimizerD = torch.optim.Adam(model.netD.parameters(), lr=args.optim_lr, betas=(0.0, 0.999))

    start_epoch = 0
    run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        args=args,
        start_epoch=start_epoch,
    )

    return


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()
