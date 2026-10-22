#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_ep275_low_lr3e5_to350_20260819"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="runs/gf_wu_full_17k_lr3e4_20260812/checkpoint_epoch_0275.pt"

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# This is a low-LR refinement branch, not a duplicate of the completed
# epoch-300--400 restart at 3e-4. The epoch-275 weights are retained while a
# fresh AdamW state and a 3-epoch warmup start a 75-epoch 3e-5 cosine cycle.
/home/m348911/miniconda3/envs/radoncAI/bin/python -u GF_RL_train.py \
  --distributed \
  --global_group_advantages \
  --checkpoint "${checkpoint}" \
  --reset_optimizer \
  --reset_best_acc \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 100 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 350 \
  --val_every 5 \
  --batch_size 4 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 3e-5 \
  --glance_lr 3e-5 \
  --warmup_epochs 3 \
  --wandb_project pancancer-ct \
  --wandb_name gf_ep275_low_lr3e5_to350_20260819 \
  2>&1 | tee -a "${run_dir}/train.log"
