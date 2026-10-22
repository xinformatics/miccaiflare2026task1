#!/usr/bin/env bash
set -euo pipefail

# Controlled higher-LR Focus escape test from the immutable epoch-280
# bounding-box champion. Both arms preserve the successful partial-label-safe
# objective and frozen Glance; only peak Focus LR changes. Each GPU receives the
# same effective global batch (8) used by the champion's 2x4-GPU run.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
lock="runs/dual_bbox_lr_escape_20260821.lock"

exec 9>"${lock}"
if ! flock -n 9; then
  echo "dual bbox LR escape experiment is already active" >&2
  exit 1
fi
[[ -s "${checkpoint}" ]]
[[ -s "${cache_root}/PREPROCESSING_COMPLETE.json" ]]
mkdir -p "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm() {
  local gpu="$1" lr="$2" tag="$3"
  local run_dir="runs/gf_bbox_escape_${tag}_ep280_5ep_20260821"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" \
    --start_epoch_override 280 --reset_optimizer --reset_best_acc \
    --freeze_glance --partial_label_teacher --bbox_partial_label \
    --bbox_context_mm 16 --pseudo_positive_threshold 0.95 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 1 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 2 --optim_lr "${lr}" \
    --warmup_epochs 1 --boundary_weight 0.15 \
    --tversky_alpha 0.3 --tversky_beta 0.7 \
    --wandb_project pancancer-ct --wandb_name "gf_bbox_escape_${tag}_ep280_5ep_20260821" \
    > "${run_dir}/launcher.log" 2>&1
}

run_arm 0 1e-4 lr1e4 & arm0=$!
run_arm 1 3e-4 lr3e4 & arm1=$!
status=0
wait "${arm0}" || status=$?
wait "${arm1}" || status=$?
exit "${status}"
