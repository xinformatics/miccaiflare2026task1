#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_screen_full_unfreeze_20260811_110459"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${CHECKPOINT:-${run_dir}/checkpoint_epoch_0118.pt}"

if [[ ! -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]]; then
  echo "Offline preprocessing is incomplete: ${cache_root}" >&2
  exit 1
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "${run_dir}" "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID="${WANDB_RUN_ID:-azay364k}"

# This is an exact continuation: deliberately omit --reset_optimizer so the
# numbered checkpoint restores AdamW and cosine-scheduler state.
/home/m348911/miniconda3/envs/radoncAI/bin/python GF_RL_train.py \
  --distributed \
  --checkpoint "${checkpoint}" \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 126 \
  --val_every 1 \
  --batch_size 1 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 1e-5 \
  --glance_lr 1e-6 \
  --partial_label_teacher \
  --pseudo_positive_threshold 0.95 \
  --pseudo_negative_threshold 0.05 \
  --pseudo_positive_weight 0.25 \
  --teacher_consistency_weight 0.15 \
  --boundary_weight 0.15 \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --warmup_epochs 1 \
  --wandb_project pancancer-ct \
  --wandb_name gf_screen_full_unfreeze_20260811_110459 \
  2>&1 | tee -a "${run_dir}/train.log"
