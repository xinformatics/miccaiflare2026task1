#!/usr/bin/env bash
set -euo pipefail

# Controlled appearance-robustness experiment queued after the current hybrid
# submission archives are safely materialized. Both arms use the same seven
# documentation-verified complete-label cohorts and the same random seed. The
# only arm difference is whether cached chest crops are remapped from their
# historical [-900,650] HU window to the deployed [-175,250] HU window.
# This controller performs public-only evaluation and never uploads anything.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/complete_multilesion_cohorts_verified7_20260822.json"
priority="${root}/configs/multilesion_source_priority04_verified7_20260822.json"
queue_marker="${root}/public_sweeps/gf_a069_bbox_pu_deployed_20260822/PUBLIC_ONLY_COMPLETE.txt"
arm_aug="gf_v7_aug_ep280_lr1e5_5ep_20260822"
arm_harmonized="gf_v7_aug_charm_ep280_lr1e5_5ep_20260822"
deployed_output="${root}/public_sweeps/gf_v7_appearance_deployed_20260822"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${complete_manifest}" "${priority}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_aug}" "${root}/runs/${arm_harmonized}" \
  "${cache_root}/tmp" "${cache_root}/component_uniform_v2"
exec 9>"${root}/runs/run_dual_verified7_appearance_after_queue_20260822.lock"
flock -n 9 || { echo "verified7 appearance experiment is already active" >&2; exit 1; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Do not compete with the current training, public audits, hidden archive
# materialization, or the queued alpha-0.69 replication. Its durable public
# marker is written only after that single-GPU experiment ends.
while [[ ! -s "${queue_marker}" ]]; do
  printf '%s waiting for alpha-0.69 public evaluation completion\n' "$(date -Ins)"
  sleep 30
done

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
  local gpu="$1" arm="$2" harmonize="$3"
  local run_dir="${root}/runs/${arm}"
  if [[ -s "${run_dir}/checkpoint_epoch_0285.pt" \
      && -s "${run_dir}/model_final.pt" ]]; then
    echo "${arm}: training already complete"
    return
  fi
  local appearance_args=(
    --enable_scale_intensity_aug
    --enable_contrast_aug
    --RandScaleIntensityd_prob 0.10
    --RandAdjustContrastd_prob 0.20
    --contrast_gamma_min 0.70
    --contrast_gamma_max 1.50
  )
  [[ "${harmonize}" == 1 ]] && appearance_args+=(--harmonize_cached_chest_window)
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_glance \
    --partial_label_teacher --bbox_partial_label --bbox_context_mm 16 \
    --hybrid_cohort_label_loss \
    --complete_label_cohorts_json "${complete_manifest}" \
    --pseudo_positive_threshold 0.95 --teacher_consistency_weight 0.15 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --optim_lr 1e-5 --warmup_epochs 1 \
    --boundary_weight 0.15 --tversky_alpha 0.3 --tversky_beta 0.7 \
    --cohort_priority_json "${priority}" \
    --component_uniform_crop --component_crop_size_power 0.5 \
    --component_crop_points_per_lesion 64 \
    --component_crop_cache_dir "${cache_root}/component_uniform_v2" \
    --seed 20260822 "${appearance_args[@]}" \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_aug}" 0 & pid_aug=$!
train_arm 1 "${arm_harmonized}" 1 & pid_harmonized=$!
status=0
wait "${pid_aug}" || status=$?
wait "${pid_harmonized}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

# Sparse trajectory audit with the exact champion deployment recipe.
ARM_A="${arm_aug}" ARM_B="${arm_harmonized}" EVAL_EPOCHS="281 283 285" \
  FOCUS_THRESHOLDS="0.40 0.50" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_bbox_lr_escape_public_20260821.sh"
