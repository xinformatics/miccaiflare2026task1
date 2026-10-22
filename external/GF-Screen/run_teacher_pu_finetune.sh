#!/usr/bin/env bash
set -euo pipefail

run_name="${RUN_NAME:-gf_screen_teacher_pu_boundary_$(date +%Y%m%d_%H%M%S)}"
run_dir="runs/${run_name}"
checkpoint="${CHECKPOINT:-pretrained/model_final.pt}"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
if [[ ! -f "${cache_root}/PREPROCESSING_COMPLETE.json" ]]; then
  echo "Offline preprocessing is incomplete: ${cache_root}" >&2
  exit 1
fi
mkdir -p "${run_dir}" "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Frozen Glance + frozen Swin encoder. The original Focus model acts as an
# anchor/teacher so missing lesions are mined rather than punished as negatives.
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
  --max_epochs 105 \
  --val_every 1 \
  --batch_size 2 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 3e-5 \
  --glance_lr 0 \
  --freeze_glance \
  --freeze_focus_encoder \
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
  --wandb_name "${run_name}" \
  2>&1 | tee -a "${run_dir}/train.log"
