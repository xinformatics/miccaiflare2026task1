#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_bbox_ep280_component16_lr1e5_5ep_20260819"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]]; then
  echo "Offline preprocessing is incomplete: ${cache_root}" >&2
  exit 1
fi

mkdir -p "${run_dir}" "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Geometry-only continuation from the immutable epoch-280 champion. Every
# annotated 26-connected component receives its own 16 mm expanded box; the
# union does not fill the bridge between distant components. Teacher positives
# are still ignored, but explicit consistency is disabled to isolate geometry.
/home/m348911/miniconda3/envs/radoncAI/bin/python -u GF_RL_train.py \
  --distributed \
  --checkpoint "${checkpoint}" \
  --start_epoch_override 280 \
  --reset_optimizer \
  --reset_best_acc \
  --freeze_glance \
  --partial_label_teacher \
  --bbox_partial_label \
  --component_boxes \
  --bbox_context_mm 16 \
  --pseudo_positive_threshold 0.95 \
  --teacher_consistency_weight 0 \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 285 \
  --val_every 1 \
  --batch_size 4 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 1e-5 \
  --warmup_epochs 1 \
  --boundary_weight 0.15 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --wandb_project pancancer-ct \
  --wandb_name gf_bbox_ep280_component16_lr1e5_5ep_20260819 \
  2>&1 | tee -a "${run_dir}/train.log"
