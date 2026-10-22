#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_wu_full_17k_lr3e4_20260812"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${CHECKPOINT:-${run_dir}/checkpoint_epoch_0155.pt}"

export TMPDIR="${cache_root}/tmp"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/m348911/miniconda3/envs/radoncAI/bin/python -u GF_RL_train.py \
  --distributed \
  --global_group_advantages \
  --checkpoint "${checkpoint}" \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 300 \
  --val_every 5 \
  --batch_size 4 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 3e-4 \
  --glance_lr 3e-4 \
  --warmup_epochs 5 \
  --wandb_project pancancer-ct \
  --wandb_name gf_wu_full_17k_lr3e4_resume0155_20260812 \
  2>&1 | tee -a "${run_dir}/train.log"
