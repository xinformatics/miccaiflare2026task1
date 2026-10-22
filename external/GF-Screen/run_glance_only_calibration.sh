#!/usr/bin/env bash
set -euo pipefail

python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
common=(
  --checkpoint "${checkpoint}"
  --reset_optimizer
  --reset_best_acc
  --data_dir .
  --abdomen_json jsons/local/abdomen.json
  --chest_json jsons/local/chest.json
  --healthy_json jsons/local/healthy.json
  --healthy_repeat 100
  --cache_dir "${cache_root}"
  --max_epochs 112
  --val_every 1
  --batch_size 1
  --sw_batch_size 2
  --samples_per_epoch 2730
  --workers 2
  --optim_lr 0
  --warmup_epochs 1
  --wandb_project pancancer-ct
)

conservative_run="runs/gf_epoch0105_glance_lr1e7_20260811"
moderate_run="runs/gf_epoch0105_glance_lr5e7_20260811"
mkdir -p "${conservative_run}" "${moderate_run}" "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GPU 0: Glance-only conservative calibration (1e-7)"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u GF_RL_train.py \
  "${common[@]}" \
  --logdir "${conservative_run}" \
  --glance_lr 1e-7 \
  --wandb_name gf_epoch0105_glance_lr1e7_20260811 \
  > "${conservative_run}/train.log" 2>&1 &
pid_conservative=$!

echo "Starting GPU 1: Glance-only moderate calibration (5e-7)"
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u GF_RL_train.py \
  "${common[@]}" \
  --logdir "${moderate_run}" \
  --glance_lr 5e-7 \
  --wandb_name gf_epoch0105_glance_lr5e7_20260811 \
  > "${moderate_run}/train.log" 2>&1 &
pid_moderate=$!

status=0
wait "${pid_conservative}" || status=$?
wait "${pid_moderate}" || status=$?
exit "${status}"
