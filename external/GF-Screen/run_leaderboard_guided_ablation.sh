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
  --freeze_glance
  --partial_label_teacher
  --pseudo_negative_threshold 0.05
  --warmup_epochs 1
  --wandb_project pancancer-ct
)

decoder_run="runs/gf_epoch0105_precision_decoder_20260811"
balanced_run="runs/gf_epoch0105_balanced_focus_20260811"
mkdir -p "${decoder_run}" "${balanced_run}" "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting GPU 0: precision-focused decoder/head adaptation"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u GF_RL_train.py \
  "${common[@]}" \
  --logdir "${decoder_run}" \
  --freeze_focus_encoder \
  --optim_lr 5e-6 \
  --pseudo_positive_threshold 0.98 \
  --pseudo_positive_weight 0.10 \
  --confident_negative_weight 0.25 \
  --teacher_consistency_weight 0.25 \
  --boundary_weight 0.25 \
  --tversky_alpha 0.65 \
  --tversky_beta 0.35 \
  --wandb_name gf_epoch0105_precision_decoder_20260811 \
  > "${decoder_run}/train.log" 2>&1 &
pid_decoder=$!

echo "Starting GPU 1: balanced full-Focus adaptation"
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u GF_RL_train.py \
  "${common[@]}" \
  --logdir "${balanced_run}" \
  --optim_lr 2e-6 \
  --pseudo_positive_threshold 0.97 \
  --pseudo_positive_weight 0.15 \
  --confident_negative_weight 0.20 \
  --teacher_consistency_weight 0.25 \
  --boundary_weight 0.25 \
  --tversky_alpha 0.50 \
  --tversky_beta 0.50 \
  --wandb_name gf_epoch0105_balanced_focus_20260811 \
  > "${balanced_run}/train.log" 2>&1 &
pid_balanced=$!

status=0
wait "${pid_decoder}" || status=$?
wait "${pid_balanced}" || status=$?
exit "${status}"
