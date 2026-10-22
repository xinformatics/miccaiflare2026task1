#!/usr/bin/env bash
set -euo pipefail

# Detection-recall experiment: component-balanced lesion crops plus bounded
# in-body background crops for teacher-safe Glance training. Both arms use the
# same appearance/domain harmonization; only the cancer positive:negative crop
# ratio differs (4:1 versus 1:1).

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/complete_multilesion_cohorts_verified7_20260822.json"
priority="${root}/configs/multilesion_source_priority04_verified7_20260822.json"
appearance_save_marker="${root}/public_sweeps/gf_v7_appearance_saved_20260822/SAVE_ONLY_COMPLETE.txt"
arm_recall="gf_glance_cbg_p4n1_ep280_5ep_20260822"
arm_balanced="gf_glance_cbg_p1n1_ep280_5ep_20260822"
deployed_output="${root}/public_sweeps/gf_glance_component_bg_deployed_20260822"
component_cache="${root}/cache/component_uniform_v3_bg4096"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${complete_manifest}" "${priority}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_recall}" "${root}/runs/${arm_balanced}" \
  "${cache_root}/tmp" "${component_cache}"
exec 9>"${root}/runs/run_dual_component_bg_glance_after_appearance_20260822.lock"
flock -n 9 || { echo "component-background Glance experiment already active" >&2; exit 1; }

for marker in "${appearance_save_marker}"; do
  while [[ ! -s "${marker}" ]]; do
    printf '%s waiting for %s\n' "$(date -Ins)" "${marker}"
    sleep 30
  done
done

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0)
  used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

train_arm() {
  local gpu="$1" arm="$2" positive="$3" negative="$4"
  local run_dir="${root}/runs/${arm}"
  if [[ -s "${run_dir}/checkpoint_epoch_0285.pt" \
      && -s "${run_dir}/model_final.pt" ]]; then
    echo "${arm}: training already complete"
    return
  fi
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_focus \
    --partial_label_teacher --glance_teacher_labels \
    --hybrid_cohort_glance_labels \
    --complete_label_cohorts_json "${complete_manifest}" \
    --glance_rl_weight 0 --glance_classification_weight 1 \
    --glance_pseudo_positive_threshold 0.95 \
    --glance_pseudo_negative_threshold 0.05 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
    --cancer_positive_weight "${positive}" --cancer_negative_weight "${negative}" \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --glance_lr 3e-5 \
    --warmup_epochs 1 --cohort_priority_json "${priority}" \
    --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 --component_background_points 4096 \
    --component_crop_cache_dir "${component_cache}" \
    --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --harmonize_cached_chest_window --seed 20260822 \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_recall}" 4 1 & pid_recall=$!
train_arm 1 "${arm_balanced}" 1 1 & pid_balanced=$!
status=0
wait "${pid_recall}" || status=$?
wait "${pid_balanced}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_recall}" ARM_B="${arm_balanced}" EVAL_EPOCHS="281 283 285" \
  GLANCE_THRESHOLDS="0.20 0.35 0.50" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_glance_teacher_safe_public_20260821.sh"
