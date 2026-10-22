#!/usr/bin/env bash
set -euo pipefail

run_dir="runs/gf_glance_only_ep280_hn300_lr3e5_5ep_20260820"
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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Controlled crop-gate experiment. Focus is exactly frozen, while Glance sees
# the annotated positive-lesion crops plus a stronger stream of verified
# healthy crops. A repeat of 300 makes healthy subjects about 40% of the
# sampling population instead of 18%, reducing batches with no negative crop.
/home/m348911/miniconda3/envs/radoncAI/bin/python -u GF_RL_train.py \
  --distributed \
  --checkpoint "${checkpoint}" \
  --start_epoch_override 280 \
  --reset_optimizer \
  --reset_best_acc \
  --freeze_focus \
  --global_group_advantages \
  --data_dir . \
  --abdomen_json jsons/local/abdomen.json \
  --chest_json jsons/local/chest.json \
  --healthy_json jsons/local/healthy.json \
  --healthy_repeat 300 \
  --cache_dir "${cache_root}" \
  --logdir "${run_dir}" \
  --max_epochs 285 \
  --val_every 1 \
  --batch_size 4 \
  --sw_batch_size 2 \
  --samples_per_epoch 2730 \
  --workers 2 \
  --optim_lr 3e-5 \
  --glance_lr 3e-5 \
  --warmup_epochs 1 \
  --wandb_project pancancer-ct \
  --wandb_name gf_glance_only_ep280_hn300_lr3e5_5ep_20260820 \
  2>&1 | tee -a "${run_dir}/train.log"

./evaluate_glance_only_checkpoints.sh
