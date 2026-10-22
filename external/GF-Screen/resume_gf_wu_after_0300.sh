#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_wu_full_17k_lr3e4_20260812"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${CHECKPOINT:-${run_dir}/checkpoint_epoch_0300.pt}"
reset_scheduler="${RESET_SCHEDULER:-auto}"
wandb_name="${WANDB_NAME:-gf_wu_full_17k_lr3e4_continue0300_to0400_20260813}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]]; then
  echo "Offline preprocessing is incomplete: ${cache_root}" >&2
  exit 1
fi

mkdir -p "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

scheduler_args=()
if [[ "${reset_scheduler}" == "auto" ]]; then
  # Only epoch 300 exhausted the original schedule. Later checkpoints already
  # contain the active 301--400 continuation schedule and must restore it.
  if [[ "$(basename "${checkpoint}")" == "checkpoint_epoch_0300.pt" ]]; then
    scheduler_args+=(--reset_scheduler)
  fi
elif [[ "${reset_scheduler}" == "1" || "${reset_scheduler}" == "true" ]]; then
  scheduler_args+=(--reset_scheduler)
elif [[ "${reset_scheduler}" != "0" && "${reset_scheduler}" != "false" ]]; then
  echo "RESET_SCHEDULER must be auto, true/1, or false/0" >&2
  exit 2
fi

# Epoch 300 ended the original cosine cycle at zero LR. Preserve the trained
# weights and AdamW moments, but begin a new 100-epoch warmup-cosine cycle.
# Hidden/public inference remains locked to thresholds 0.5/0.5, abdomen HU
# [-175, 250], and a 1,000 mm3 connected-component filter.
/home/m348911/miniconda3/envs/radoncAI/bin/python -u GF_RL_train.py \
  --distributed \
  --global_group_advantages \
  --checkpoint "${checkpoint}" \
  "${scheduler_args[@]}" \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 400 \
  --val_every 5 \
  --batch_size 4 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 3e-4 \
  --glance_lr 3e-4 \
  --warmup_epochs 5 \
  --wandb_project pancancer-ct \
  --wandb_name "${wandb_name}" \
  2>&1 | tee -a "${run_dir}/train.log"
