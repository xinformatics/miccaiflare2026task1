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
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from utils.optimizers import LinearWarmupCosineAnnealingLR
from GF_RL_trainer import run_training
from utils.data_utils import get_loader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from utils.utils import *
from models.GF_Screen_RL import GFScreen_RL
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

import resource
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

parser = argparse.ArgumentParser(description="Segmentation pipeline")
parser.add_argument("--name", default=None, help="the name of pre-trained model !!!!!!!!")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--pretrained_root", default='pretrained/', help="pre-trained root")
parser.add_argument("--glance_backbone", default='resnet18', help="pre-trained root")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument(
    "--partial_label_teacher_checkpoint",
    default="",
    type=str,
    help=(
        "optional immutable checkpoint supplying the frozen Focus teacher; "
        "use this when a partial-label run may resume from a newer student "
        "checkpoint"
    ),
)
parser.add_argument(
    "--start_epoch_override",
    default=None,
    type=int,
    help=(
        "override checkpoint epoch metadata for derived/interpolated weights; "
        "the value is the first epoch index to train"
    ),
)
parser.add_argument(
    "--reset_optimizer",
    action="store_true",
    help="load checkpoint weights/epoch but start fresh optimizers and LR schedules",
)
parser.add_argument(
    "--reset_scheduler",
    action="store_true",
    help="preserve checkpoint optimizer moments but start a fresh LR schedule",
)
parser.add_argument(
    "--reset_best_acc",
    action="store_true",
    help="track a new run-local champion instead of inheriting the checkpoint best score",
)
parser.add_argument("--logdir", default="./runs/logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="./data", type=str, help="dataset directory")
parser.add_argument("--cache_dir", default='./data/cache', type=str, help="dataset json file")
parser.add_argument("--abdomen_json", default='./jsons/local/abdomen.json', type=str)
parser.add_argument("--chest_json", default='./jsons/local/chest.json', type=str)
parser.add_argument("--healthy_json", default='./jsons/local/healthy.json', type=str)
parser.add_argument("--healthy_repeat", default=10, type=int)
parser.add_argument(
    "--pseudo_abdomen_json",
    default="",
    type=str,
    help=(
        "optional positive-only abdominal teacher pseudo-label manifest; "
        "entries are appended to the partial-label abdomen stream"
    ),
)
parser.add_argument(
    "--pseudo_abdomen_repeat",
    default=1,
    type=int,
    help="integer exposure multiplier for --pseudo_abdomen_json entries",
)
parser.add_argument(
    "--cancer_positive_weight",
    default=1.0,
    type=float,
    help="positive weight in cancer-scan patch sampling (legacy default: 1)",
)
parser.add_argument(
    "--cancer_negative_weight",
    default=0.0,
    type=float,
    help=(
        "negative weight in cancer-scan patch sampling; safe use requires "
        "frozen Glance and teacher-backed bounding-box supervision"
    ),
)
parser.add_argument("--use_persistent_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training")
parser.add_argument(
    "--checkpoint_every",
    default=1,
    type=int,
    help=(
        "retain a recoverable optimizer checkpoint every N completed epochs; "
        "the final epoch is always retained"
    ),
)
parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--update_epochs", default=1, type=int, help="epochs to update policy model")
parser.add_argument("--val_every", default=500, type=int, help="validation frequency")
parser.add_argument(
    "--skip_final_validation",
    action="store_true",
    help=(
        "do not force full-volume validation at the final epoch; intended for "
        "checkpoint-grid controllers that run a separate locked evaluator"
    ),
)

parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument(
    "--validation_focus_batch_size",
    default=2,
    type=int,
    help="Focus crop microbatch used only by full-volume validation",
)
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--glance_lr", default=3e-5, type=float, help="Glance learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", default=False, help="do NOT use amp for training")

parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--samples_per_epoch", default=4096, type=int,
                    help="global cases sampled per epoch; 0 uses the full corpus")
parser.add_argument(
    "--cohort_balance_power",
    default=0.0,
    type=float,
    help=(
        "capped inverse-frequency cohort sampling exponent; zero preserves "
        "the original case-uniform sampler and 0.5 applies square-root balancing"
    ),
)
parser.add_argument(
    "--cohort_balance_max_weight",
    default=5.0,
    type=float,
    help=(
        "maximum rare-to-largest cohort weight ratio before normalization; "
        "normalization preserves the lesion-versus-proxy sampling ratio"
    ),
)
parser.add_argument(
    "--cohort_priority_json",
    default="",
    type=str,
    help=(
        "optional JSON mapping exact region:source cohort names to positive "
        "sampling multipliers; lesion mass is renormalized after applying it"
    ),
)
parser.add_argument(
    "--complete_label_cohorts_json",
    default="",
    type=str,
    help=(
        "versioned JSON list of cohorts whose masks may receive full-volume "
        "loss in --hybrid_cohort_label_loss mode"
    ),
)
parser.add_argument(
    "--component_uniform_crop",
    action="store_true",
    help=(
        "sample positive crop centres uniformly across annotated connected "
        "components instead of uniformly across foreground voxels"
    ),
)
parser.add_argument(
    "--component_crop_points_per_lesion",
    default=64,
    type=int,
    help="equal number of candidate crop centres retained for each lesion component",
)
parser.add_argument(
    "--component_crop_size_power",
    default=0.0,
    type=float,
    help=(
        "component crop weighting exponent: 0 is component-uniform, 1 "
        "approximates foreground-voxel-uniform, and 0.5 is square-root balance"
    ),
)
parser.add_argument(
    "--component_crop_cache_dir",
    default="",
    type=str,
    help="optional sidecar directory for reusable component-balanced crop indices",
)
parser.add_argument(
    "--component_background_points",
    default=0,
    type=int,
    help=(
        "bounded number of deterministic body-background crop candidates per "
        "case in component-uniform mode; required when cancer negative "
        "sampling is enabled"
    ),
)
parser.add_argument(
    "--component_legacy_cache_dir",
    default="",
    type=str,
    help=(
        "optional positive-only v2 component-index directory reused when "
        "building background-enabled sidecars"
    ),
)

parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=3.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument(
    "--enable_scale_intensity_aug",
    action="store_true",
    help="opt in to the formerly configured but unused random intensity scaling",
)
parser.add_argument(
    "--enable_contrast_aug",
    action="store_true",
    help="opt in to random gamma contrast after crop extraction",
)
parser.add_argument(
    "--RandAdjustContrastd_prob",
    default=0.2,
    type=float,
    help="random gamma-contrast probability when --enable_contrast_aug is set",
)
parser.add_argument(
    "--contrast_gamma_min", default=0.7, type=float,
    help="minimum gamma for opt-in contrast augmentation",
)
parser.add_argument(
    "--contrast_gamma_max", default=1.5, type=float,
    help="maximum gamma for opt-in contrast augmentation",
)
parser.add_argument(
    "--harmonize_cached_chest_window",
    action="store_true",
    help=(
        "remap cached chest crops from the training [-900,650] window to the "
        "challenge inference [-175,250] window"
    ),
)
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="warmup_cosine, cosine_anneal")
parser.add_argument("--warmup_epochs", default=1, type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
parser.add_argument("--wandb_project", default="pancancer-ct", type=str)
parser.add_argument("--wandb_name", default=None, type=str)
parser.add_argument("--wandb_mode", default="online", choices=("online", "offline", "disabled"))
parser.add_argument("--seed", default=20260809, type=int)
parser.add_argument("--freeze_glance", action="store_true",
                    help="preserve the released Glance gate during adaptation")
parser.add_argument("--freeze_focus", action="store_true",
                    help="preserve the complete Focus segmenter and train only Glance")
parser.add_argument("--freeze_focus_encoder", action="store_true",
                    help="train only the SwinUNETR decoder and output head")
parser.add_argument("--partial_label_teacher", action="store_true",
                    help="anchor adaptation to the frozen released Focus model")
parser.add_argument(
    "--bbox_partial_label",
    action="store_true",
    help=(
        "supervise lesion-bearing Focus crops only inside an automatically "
        "expanded target-lesion box; voxels outside the box are ignored"
    ),
)
parser.add_argument(
    "--hybrid_cohort_label_loss",
    action="store_true",
    help=(
        "use full DiceCE supervision only for explicitly complete cohorts and "
        "bounding-box/teacher-safe supervision for all other cancer cohorts"
    ),
)
parser.add_argument(
    "--hybrid_cohort_glance_labels",
    action="store_true",
    help=(
        "treat annotated-negative Glance crops as trusted only for explicitly "
        "complete cohorts; retain frozen-Focus teacher gating elsewhere"
    ),
)
parser.add_argument(
    "--bbox_context_mm",
    default=16.0,
    type=float,
    help="physical context margin on every side of the annotated-lesion box",
)
parser.add_argument(
    "--component_boxes",
    action="store_true",
    help=(
        "build an independently expanded box around every annotated connected "
        "component instead of one enclosing box around all foreground"
    ),
)
parser.add_argument("--pseudo_positive_threshold", default=0.95, type=float)
parser.add_argument("--pseudo_negative_threshold", default=0.05, type=float)
parser.add_argument("--pseudo_positive_weight", default=0.25, type=float)
parser.add_argument(
    "--bbox_pseudo_positive_weight",
    default=0.0,
    type=float,
    help=(
        "additional hard-positive BCE weight for high-confidence frozen-teacher "
        "foreground outside the annotated target box; zero preserves the "
        "original bounding-box objective"
    ),
)
parser.add_argument(
    "--bbox_empty_pseudo_positive_weight",
    default=0.0,
    type=float,
    help=(
        "positive-only BCE weight for sizeable high-confidence teacher lesions "
        "found in nominally negative crops from partially labelled cancer scans"
    ),
)
parser.add_argument(
    "--bbox_empty_pseudo_positive_min_voxels",
    default=64,
    type=int,
    help="minimum high-confidence teacher foreground voxels required per empty cancer crop",
)
parser.add_argument("--confident_negative_weight", default=0.10, type=float)
parser.add_argument("--teacher_consistency_weight", default=0.15, type=float)
parser.add_argument("--boundary_weight", default=0.15, type=float)
parser.add_argument("--tversky_alpha", default=0.3, type=float)
parser.add_argument("--tversky_beta", default=0.7, type=float)
parser.add_argument(
    "--global_group_advantages",
    action="store_true",
    help="normalize GRL rewards across all DDP ranks to preserve the intended group size",
)
parser.add_argument(
    "--glance_classification_weight",
    default=0.1,
    type=float,
    help="weight of direct lesion-present crop classification in the total loss",
)
parser.add_argument(
    "--glance_rl_weight",
    default=1.0,
    type=float,
    help="weight of the Focus-reward Glance policy loss in the total loss",
)
parser.add_argument(
    "--glance_teacher_labels",
    action="store_true",
    help=(
        "train Glance with annotated positives, frozen-Focus pseudo positives, "
        "teacher-safe negatives, and ignored uncertain crops"
    ),
)
parser.add_argument("--glance_pseudo_positive_threshold", default=0.95, type=float)
parser.add_argument("--glance_pseudo_negative_threshold", default=0.05, type=float)
parser.add_argument("--glance_pseudo_positive_min_voxels", default=64, type=int)
parser.add_argument("--glance_teacher_batch_size", default=2, type=int)
parser.add_argument(
    "--glance_distillation_weight",
    default=0.0,
    type=float,
    help=(
        "KL weight that preserves the frozen starting Glance policy on crops "
        "ignored by partial-label-safe classification"
    ),
)
parser.add_argument(
    "--glance_box_localization",
    action="store_true",
    help=(
        "add a training-time six-coordinate lesion-box head to Glance; the "
        "deployed binary crop gate remains unchanged"
    ),
)
parser.add_argument(
    "--glance_box_localization_weight",
    default=0.0,
    type=float,
    help="weight of the annotation-positive Glance component-box Smooth-L1 loss",
)
parser.add_argument(
    "--glance_center_localization",
    action="store_true",
    help=(
        "add a training-time quarter-resolution dense lesion-centre head to "
        "Glance; deployment remains the released binary crop gate"
    ),
)
parser.add_argument(
    "--glance_center_localization_weight",
    default=0.0,
    type=float,
    help=(
        "weight of the partial-label-safe dense Glance centerness focal loss"
    ),
)


def main():
    args = parser.parse_args()
    if args.pseudo_abdomen_repeat < 1:
        parser.error("--pseudo_abdomen_repeat must be at least 1")
    if args.pseudo_abdomen_json and not os.path.isfile(args.pseudo_abdomen_json):
        parser.error(f"missing --pseudo_abdomen_json: {args.pseudo_abdomen_json}")
    if (
        args.partial_label_teacher_checkpoint
        and not os.path.isfile(args.partial_label_teacher_checkpoint)
    ):
        parser.error(
            "missing --partial_label_teacher_checkpoint: "
            f"{args.partial_label_teacher_checkpoint}"
        )
    if args.partial_label_teacher_checkpoint and not args.partial_label_teacher:
        parser.error(
            "--partial_label_teacher_checkpoint requires "
            "--partial_label_teacher"
        )
    if args.freeze_glance and args.freeze_focus:
        parser.error("--freeze_glance and --freeze_focus cannot be used together")
    if args.glance_classification_weight < 0 or args.glance_rl_weight < 0:
        parser.error("Glance loss weights must be non-negative")
    if args.glance_teacher_labels and not (
        args.freeze_focus and args.partial_label_teacher
    ):
        parser.error(
            "--glance_teacher_labels requires --freeze_focus, "
            "and --partial_label_teacher"
        )
    if not (
        0 <= args.glance_pseudo_negative_threshold
        < args.glance_pseudo_positive_threshold <= 1
    ):
        parser.error(
            "Glance teacher thresholds must satisfy 0 <= negative < positive <= 1"
        )
    if args.glance_pseudo_positive_min_voxels < 1:
        parser.error("--glance_pseudo_positive_min_voxels must be at least 1")
    if args.glance_teacher_batch_size < 1:
        parser.error("--glance_teacher_batch_size must be at least 1")
    if args.validation_focus_batch_size < 1:
        parser.error("--validation_focus_batch_size must be at least 1")
    if args.glance_distillation_weight < 0:
        parser.error("--glance_distillation_weight must be non-negative")
    if args.glance_distillation_weight > 0 and not (
        args.freeze_focus
        and args.glance_teacher_labels
        and args.partial_label_teacher
        and args.hybrid_cohort_glance_labels
    ):
        parser.error(
            "--glance_distillation_weight > 0 requires frozen Focus, "
            "teacher-safe Glance labels, and hybrid cohort metadata"
        )
    if args.glance_box_localization_weight < 0:
        parser.error("--glance_box_localization_weight must be non-negative")
    if args.glance_box_localization != (args.glance_box_localization_weight > 0):
        parser.error(
            "--glance_box_localization and a positive "
            "--glance_box_localization_weight must be used together"
        )
    if args.glance_box_localization and not (
        args.freeze_focus
        and args.glance_teacher_labels
        and args.partial_label_teacher
        and args.component_uniform_crop
        and args.complete_label_cohorts_json
        and args.reset_optimizer
    ):
        parser.error(
            "Glance box localization requires frozen Focus, teacher-safe Glance "
            "labels, component-uniform sampling, complete-cohort metadata, and "
            "a reset optimizer"
        )
    if args.glance_center_localization_weight < 0:
        parser.error("--glance_center_localization_weight must be non-negative")
    if args.glance_center_localization != (
        args.glance_center_localization_weight > 0
    ):
        parser.error(
            "--glance_center_localization and a positive "
            "--glance_center_localization_weight must be used together"
        )
    if args.glance_center_localization and not (
        args.freeze_focus
        and args.glance_teacher_labels
        and args.partial_label_teacher
        and args.hybrid_cohort_glance_labels
        and args.component_uniform_crop
        and args.complete_label_cohorts_json
        and args.reset_optimizer
    ):
        parser.error(
            "Glance center localization requires frozen Focus, teacher-safe "
            "Glance labels, hybrid complete-cohort metadata, component-uniform "
            "sampling, and a reset optimizer"
        )
    if args.bbox_pseudo_positive_weight < 0:
        parser.error("--bbox_pseudo_positive_weight must be non-negative")
    if args.bbox_empty_pseudo_positive_weight < 0:
        parser.error("--bbox_empty_pseudo_positive_weight must be non-negative")
    if args.bbox_empty_pseudo_positive_min_voxels < 1:
        parser.error("--bbox_empty_pseudo_positive_min_voxels must be at least 1")
    if args.bbox_empty_pseudo_positive_weight > 0 and not (
        args.freeze_glance
        and args.bbox_partial_label
        and args.partial_label_teacher
        and args.cancer_negative_weight > 0
    ):
        parser.error(
            "--bbox_empty_pseudo_positive_weight > 0 requires --freeze_glance, "
            "--bbox_partial_label, --partial_label_teacher, and "
            "--cancer_negative_weight > 0"
        )
    if args.cohort_balance_power < 0:
        parser.error("--cohort_balance_power must be non-negative")
    if args.cohort_balance_max_weight < 1:
        parser.error("--cohort_balance_max_weight must be at least 1")
    if args.cancer_positive_weight < 0 or args.cancer_negative_weight < 0:
        parser.error("cancer patch-sampling weights must be non-negative")
    if args.cancer_positive_weight + args.cancer_negative_weight <= 0:
        parser.error("at least one cancer patch-sampling weight must be positive")
    if args.component_crop_points_per_lesion < 1:
        parser.error("--component_crop_points_per_lesion must be at least 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint_every must be at least 1")
    if not 0 <= args.component_crop_size_power <= 1:
        parser.error("--component_crop_size_power must be in [0, 1]")
    if args.component_background_points < 0:
        parser.error("--component_background_points must be non-negative")
    if (
        args.component_uniform_crop
        and args.cancer_negative_weight > 0
        and args.component_background_points < 1
    ):
        parser.error(
            "component-uniform cancer negative sampling requires "
            "--component_background_points >= 1"
        )
    if args.complete_label_cohorts_json and not (
        args.hybrid_cohort_label_loss or args.hybrid_cohort_glance_labels
    ):
        parser.error(
            "--complete_label_cohorts_json requires a hybrid cohort loss mode"
        )
    if args.hybrid_cohort_label_loss and not (
        args.freeze_glance
        and args.bbox_partial_label
        and args.partial_label_teacher
        and args.complete_label_cohorts_json
    ):
        parser.error(
            "--hybrid_cohort_label_loss requires --freeze_glance, "
            "--bbox_partial_label, --partial_label_teacher, and "
            "--complete_label_cohorts_json"
        )
    if args.hybrid_cohort_glance_labels and not (
        args.freeze_focus
        and args.glance_teacher_labels
        and args.partial_label_teacher
        and args.complete_label_cohorts_json
    ):
        parser.error(
            "--hybrid_cohort_glance_labels requires --freeze_focus, "
            "--glance_teacher_labels, --partial_label_teacher, and "
            "--complete_label_cohorts_json"
        )
    safe_focus_negative_training = (
        args.freeze_glance and args.bbox_partial_label and args.partial_label_teacher
    )
    safe_glance_negative_training = (
        args.freeze_focus
        and args.glance_teacher_labels
        and args.partial_label_teacher
    )
    if args.pseudo_abdomen_json and not (
        safe_focus_negative_training or safe_glance_negative_training
    ):
        parser.error(
            "--pseudo_abdomen_json requires either positive-only Focus "
            "training (--freeze_glance, --bbox_partial_label, "
            "--partial_label_teacher) or teacher-safe Glance training "
            "(--freeze_focus, --glance_teacher_labels, "
            "--partial_label_teacher)"
        )
    if args.cancer_negative_weight > 0 and not (
        safe_focus_negative_training or safe_glance_negative_training
    ):
        parser.error(
            "--cancer_negative_weight > 0 requires either teacher-safe Focus "
            "training (--freeze_glance, --bbox_partial_label, "
            "--partial_label_teacher) or teacher-safe Glance training "
            "(--freeze_focus, --glance_teacher_labels, "
            "--partial_label_teacher)"
        )
    args.amp = not args.noamp
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    print(args)
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    model = GFScreen_RL(args)

    if args.squared_dice:
        dice_loss = DiceCELoss(
            to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    checkpoint = None
    if args.checkpoint is not None:
        # Training checkpoints are created locally by save_checkpoint and
        # include trusted optimizer/scheduler metadata with NumPy scalars.
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "epoch" in checkpoint:
            # Stored epoch is zero-based and complete; resume at the next one.
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]

        model_dict = checkpoint["state_dict"]
        if args.glance_box_localization or args.glance_center_localization:
            incompatible = model.load_state_dict(model_dict, strict=False)
            allowed_missing = set()
            if args.glance_box_localization:
                allowed_missing.update(
                    {
                        "glance_model.box_layer.weight",
                        "glance_model.box_layer.bias",
                    }
                )
            if args.glance_center_localization:
                allowed_missing.update(
                    {
                        "glance_model.center_layer.weight",
                        "glance_model.center_layer.bias",
                    }
                )
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Glance auxiliary-localization checkpoint mismatch: "
                    f"missing={incompatible.missing_keys} "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            if set(incompatible.missing_keys):
                print(
                    "Initialized new Glance auxiliary-localization tensors",
                    sorted(incompatible.missing_keys),
                )
        else:
            model.load_state_dict(model_dict, strict=True)

        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
        if args.start_epoch_override is not None:
            print(
                "Overriding checkpoint-derived start epoch",
                start_epoch,
                "with",
                args.start_epoch_override,
            )
            start_epoch = args.start_epoch_override
        if args.reset_best_acc:
            print("Resetting run-local best accuracy from", best_acc, "to 0")
            best_acc = 0.0

    if args.freeze_glance:
        for parameter in model.glance_model.parameters():
            parameter.requires_grad_(False)
        print("Glance gate frozen")
    if args.freeze_focus and args.freeze_focus_encoder:
        raise ValueError("--freeze_focus and --freeze_focus_encoder are mutually exclusive")
    if args.freeze_focus:
        for parameter in model.segmentation_model.parameters():
            parameter.requires_grad_(False)
        print("Focus segmenter frozen; Glance-only adaptation enabled")
    if args.freeze_focus_encoder:
        frozen_prefixes = ("swinViT.", "encoder1.", "encoder2.", "encoder3.", "encoder4.", "encoder10.")
        for name, parameter in model.segmentation_model.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad_(False)
        trainable_focus = sum(p.numel() for p in model.segmentation_model.parameters() if p.requires_grad)
        print("Focus encoder frozen; trainable decoder/head parameters", trainable_focus)

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    core_model = model.module if args.distributed else model
    teacher_segmentation = None
    if (
        args.partial_label_teacher
        and not (args.freeze_focus and args.glance_teacher_labels)
    ):
        import copy
        teacher_segmentation = copy.deepcopy(core_model.segmentation_model)
        if args.partial_label_teacher_checkpoint:
            teacher_checkpoint = torch.load(
                args.partial_label_teacher_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            teacher_full_state = teacher_checkpoint.get("state_dict")
            if not isinstance(teacher_full_state, dict):
                raise ValueError(
                    "partial-label teacher checkpoint has no state_dict"
                )
            prefixes = ("segmentation_model.", "module.segmentation_model.")
            teacher_state = {
                key[len(prefix):]: value
                for key, value in teacher_full_state.items()
                for prefix in prefixes
                if key.startswith(prefix)
            }
            if not teacher_state:
                raise ValueError(
                    "partial-label teacher checkpoint has no Focus tensors"
                )
            teacher_segmentation.load_state_dict(teacher_state, strict=True)
            print(
                "Frozen Focus teacher loaded from",
                args.partial_label_teacher_checkpoint,
            )
        teacher_segmentation.requires_grad_(False)
        teacher_segmentation.eval()
        teacher_segmentation.cuda(args.gpu)
        print("Frozen released Focus teacher enabled for missing-lesion mining")
    elif args.partial_label_teacher and args.freeze_focus and args.glance_teacher_labels:
        print("Frozen Focus outputs enabled as teacher-safe Glance crop labels")
    focus_parameters = [p for p in core_model.segmentation_model.parameters() if p.requires_grad]
    glance_parameters = [p for p in core_model.glance_model.parameters() if p.requires_grad]
    optimizer_seg = (
        torch.optim.AdamW(focus_parameters, lr=args.optim_lr, weight_decay=args.reg_weight)
        if focus_parameters else None
    )
    optimizer_glance = (
        torch.optim.AdamW(glance_parameters, lr=args.glance_lr, weight_decay=args.reg_weight)
        if glance_parameters else None
    )

    if (
        checkpoint is not None
        and not args.reset_optimizer
        and "optimizer_seg" in checkpoint
        and optimizer_seg is not None
    ):
        optimizer_seg.load_state_dict(checkpoint["optimizer_seg"])
    if (
        checkpoint is not None
        and not args.reset_optimizer
        and "optimizer_glance" in checkpoint
        and optimizer_glance is not None
    ):
        optimizer_glance.load_state_dict(checkpoint["optimizer_glance"])

    if args.lrschedule == "warmup_cosine":
        # max_epochs is an absolute epoch index when resuming the released
        # epoch-99 checkpoint; schedule only across the new adaptation span.
        max_steps = max(1, args.max_epochs - start_epoch) * len(loader[0])
        warmup_steps = args.warmup_epochs * len(loader[0])
        scheduler_seg = (
            LinearWarmupCosineAnnealingLR(
                optimizer_seg, warmup_epochs=warmup_steps, max_epochs=max_steps
            )
            if optimizer_seg is not None else None
        )
        scheduler_glance = (
            LinearWarmupCosineAnnealingLR(optimizer_glance, warmup_epochs=warmup_steps, max_epochs=max_steps)
            if optimizer_glance is not None else None
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler_seg = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_seg, T_max=args.max_epochs)
            if optimizer_seg is not None else None
        )
        scheduler_glance = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_glance, T_max=args.max_epochs)
            if optimizer_glance is not None else None
        )
        if args.checkpoint is not None:
            if scheduler_seg is not None:
                scheduler_seg.step(epoch=start_epoch)
            if scheduler_glance is not None:
                scheduler_glance.step(epoch=start_epoch)
    else:
        scheduler_seg = scheduler_glance = None

    if (
        checkpoint is not None
        and not args.reset_optimizer
        and not args.reset_scheduler
        and "scheduler_seg" in checkpoint
        and scheduler_seg is not None
    ):
        scheduler_seg.load_state_dict(checkpoint["scheduler_seg"])
        # _LRScheduler construction initializes the optimizer LR at the start
        # of warmup. load_state_dict restores scheduler counters but does not
        # restore optimizer param-group LRs, which otherwise leaves a resumed
        # run permanently at zero LR for this chainable cosine implementation.
        for param_group, lr in zip(optimizer_seg.param_groups, scheduler_seg.get_last_lr()):
            param_group["lr"] = lr
    if (
        checkpoint is not None
        and not args.reset_optimizer
        and not args.reset_scheduler
        and "scheduler_glance" in checkpoint
        and scheduler_glance is not None
    ):
        scheduler_glance.load_state_dict(checkpoint["scheduler_glance"])
        for param_group, lr in zip(optimizer_glance.param_groups, scheduler_glance.get_last_lr()):
            param_group["lr"] = lr

    run_training(
        model,
        loader[0],
        loader[1],
        (optimizer_seg, optimizer_glance),
        dice_loss,
        dice_acc,
        args,
        scheduler=(scheduler_seg, scheduler_glance),
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
        teacher_segmentation=teacher_segmentation,
        best_acc=best_acc,
    )
    if args.distributed:
        # Keep nonzero ranks alive until rank 0 has flushed checkpoints,
        # TensorBoard, and W&B, then release NCCL explicitly.
        dist.barrier()
        dist.destroy_process_group()


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
