#!/usr/bin/env bash
set -euo pipefail

# Dense quarter-resolution centerness supervision inspired by LENS.  This
# controlled Glance-only experiment starts from the strongest pre-box CBG
# checkpoint, preserves verified complete/healthy negatives, and never treats
# unannotated regions in partial cohorts as background.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
upstream="${root}/public_sweeps/gf_glance_cbg_box_saved_20260822/SAVE_ONLY_COMPLETE.txt"
complete_manifest="${root}/configs/complete_multilesion_cohorts_verified7_20260822.json"
priority="${root}/configs/multilesion_source_priority04_verified7_20260822.json"
component_cache="${root}/cache/component_uniform_v3_bg4096"
arm_low="gf_glance_cbg_center_w025_ep283_2ep_20260822"
arm_high="gf_glance_cbg_center_w075_ep283_2ep_20260822"
deployed="${root}/public_sweeps/gf_glance_cbg_center_deployed_20260822"

for required in "${checkpoint}" "${upstream}" "${complete_manifest}" \
    "${priority}" "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  while [[ ! -s "${required}" ]]; do
    printf '%s waiting for %s\n' "$(date -Ins)" "${required}"
    sleep 10
  done
done
mkdir -p "${root}/runs/${arm_low}" "${root}/runs/${arm_high}" \
  "${component_cache}" "${cache_root}/tmp"
exec 9>"${root}/runs/run_dual_cbg_center_localization_after_box_20260822.lock"
flock -n 9 || { echo "CBG center-localization experiment already active" >&2; exit 1; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 10
done

train_arm() {
  local gpu="$1" arm="$2" weight="$3" run_dir="${root}/runs/$2"
  [[ -s "${run_dir}/checkpoint_epoch_0285.pt" && -s "${run_dir}/model_final.pt" ]] && return
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 283 \
    --reset_optimizer --reset_best_acc --freeze_focus \
    --partial_label_teacher --glance_teacher_labels --hybrid_cohort_glance_labels \
    --complete_label_cohorts_json "${complete_manifest}" \
    --glance_rl_weight 0 --glance_classification_weight 1 \
    --glance_center_localization --glance_center_localization_weight "${weight}" \
    --glance_pseudo_positive_threshold 0.95 --glance_pseudo_negative_threshold 0.05 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
    --glance_distillation_weight 0.25 --update_epochs 999 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 2 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --glance_lr 1e-5 --warmup_epochs 1 \
    --cohort_priority_json "${priority}" \
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
train_arm 0 "${arm_low}" 0.25 & pid0=$!
train_arm 1 "${arm_high}" 0.75 & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_low}" ARM_B="${arm_high}" EVAL_EPOCHS="285" \
  GLANCE_THRESHOLD="0.50" CENTER_THRESHOLDS="off 0.90 0.97 0.99" \
  DEPLOYED_OUTPUT="${deployed}" \
  bash "${root}/evaluate_dual_glance_center_gate_public_20260822.sh"
"${python_bin}" "${root}/audit_center_small_lesion_recall.py" \
  --summary "${deployed}/summary.tsv" \
  --anchor-audit "${root}/analysis/lesion_error_champion_new2k_s25_20260822/summary.json" \
  --labels "${root}/../../validation/Validation-Public-Labels" \
  --audit-script "${root}/analyze_lesion_errors.py" \
  --output "${deployed}/small_lesion_audit"
bash "${root}/save_cbg_center_candidate_20260822.sh"
