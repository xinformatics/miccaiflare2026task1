#!/usr/bin/env bash
set -euo pipefail

# Full-corpus Focus adaptation under the corrected challenge contract: every
# cancer mask is partial, only annotation-positive lesion crops are sampled,
# and loss outside the expanded target box is ignored/teacher anchored.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
contract="${root}/configs/challenge_train_labeled_all_partial_20260822.json"
empty_healthy="${root}/jsons/local/no_proxy_healthy.json"
# Keep this lower-priority branch behind candidate refinement, corrected CBG
# box localization, and deployed centerness so two-GPU watchers cannot race.
upstream="${root}/public_sweeps/gf_glance_cbg_center_saved_20260822/SAVE_ONLY_COMPLETE.txt"
component_cache="${root}/cache/component_uniform_v3_bg4096"
arm_low="gf_focus_allpartial_pos_lr1e5_ep280_3ep_20260822"
arm_high="gf_focus_allpartial_pos_lr3e5_ep280_3ep_20260822"
deployed_output="${root}/public_sweeps/gf_focus_allpartial_deployed_20260822"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${contract}" "${empty_healthy}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_low}" "${root}/runs/${arm_high}" \
  "${cache_root}/tmp" "${component_cache}"
exec 9>"${root}/runs/run_dual_focus_allpartial_after_boxloc_20260822.lock"
flock -n 9 || { echo "all-partial Focus experiment already active" >&2; exit 1; }
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for CBG centerness save-only completion\n' "$(date -Ins)"
  sleep 30
done

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done

train_arm() {
  local gpu="$1" arm="$2" lr="$3" run_dir="${root}/runs/$2"
  [[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]] && return
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_glance \
    --partial_label_teacher --bbox_partial_label --bbox_context_mm 16 \
    --pseudo_positive_threshold 0.95 --pseudo_negative_threshold 0.05 \
    --teacher_consistency_weight 0.15 --boundary_weight 0.15 \
    --tversky_alpha 0.3 --tversky_beta 0.7 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json "${empty_healthy}" \
    --healthy_repeat 1 --cancer_positive_weight 1 --cancer_negative_weight 0 \
    --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 283 --val_every 3 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 20000 --workers 1 --optim_lr "${lr}" --warmup_epochs 1 \
    --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 \
    --component_crop_cache_dir "${component_cache}" \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --harmonize_cached_chest_window --seed 20260822 \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_low}" 1e-5 & pid_low=$!
train_arm 1 "${arm_high}" 3e-5 & pid_high=$!
status=0
wait "${pid_low}" || status=$?
wait "${pid_high}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_low}" ARM_B="${arm_high}" EVAL_EPOCHS="281 282 283" \
  FOCUS_THRESHOLDS="0.40 0.50 0.60" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_bbox_lr_escape_public_20260821.sh"
bash "${root}/save_focus_allpartial_candidate_20260822.sh"
