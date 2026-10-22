#!/usr/bin/env bash
set -euo pipefail

# Orthogonal crop-recall experiment from the immutable epoch-280 champion.
# Focus is frozen.  Both arms train Glance with annotation positives, frozen-
# Focus pseudo positives, teacher-safe negatives, and ignored uncertain crops.
# The only difference is whether cancer scans also sample nominally negative
# regions, where missing annotations are explicitly protected by the teacher.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
lock="runs/dual_glance_teacher_safe_20260821.lock"

exec 9>"${lock}"
if ! flock -n 9; then
  echo "dual teacher-safe Glance experiment is already active" >&2
  exit 1
fi
[[ -s "${checkpoint}" ]]
[[ -s "${cache_root}/PREPROCESSING_COMPLETE.json" ]]
mkdir -p "${cache_root}/tmp"
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_arm() {
  local gpu="$1" negative_weight="$2" tag="$3"
  local run_dir="runs/gf_glance_safe_${tag}_ep280_5ep_20260821"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" \
    --start_epoch_override 280 --reset_optimizer --reset_best_acc \
    --freeze_focus --partial_label_teacher --glance_teacher_labels \
    --glance_rl_weight 0 --glance_classification_weight 1 \
    --glance_pseudo_positive_threshold 0.95 \
    --glance_pseudo_negative_threshold 0.05 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
    --cancer_positive_weight 1 --cancer_negative_weight "${negative_weight}" \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 1 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 2 --glance_lr 1e-4 \
    --warmup_epochs 1 --wandb_project pancancer-ct \
    --wandb_name "gf_glance_safe_${tag}_ep280_5ep_20260821" \
    > "${run_dir}/launcher.log" 2>&1
}

run_arm 0 0 posonly & arm0=$!
run_arm 1 1 posneg & arm1=$!
status=0
wait "${arm0}" || status=$?
wait "${arm1}" || status=$?
exit "${status}"
