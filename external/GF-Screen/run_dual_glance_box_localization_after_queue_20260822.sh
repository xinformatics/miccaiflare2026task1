#!/usr/bin/env bash
set -euo pipefail

# Two-arm auxiliary box-localization experiment. It waits for every currently
# serialized public audit, then trains one Glance model per GPU. Focus and all
# deployed postprocessing stay immutable. No hidden inference or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/challenge_train_labeled_all_partial_20260822.json"
empty_healthy="${root}/jsons/local/no_proxy_healthy.json"
upstream="${root}/public_sweeps/bbox_wise_and_anatomy_queue_20260822_COMPLETE.txt"
component_cache="${root}/cache/component_uniform_v3_bg4096"
arm_low="gf_glance_boxloc_w005_allpartial_ep280_3ep_20260822"
arm_high="gf_glance_boxloc_w015_allpartial_ep280_3ep_20260822"
deployed_output="${root}/public_sweeps/gf_glance_boxloc_deployed_20260822"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${complete_manifest}" "${empty_healthy}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_low}" "${root}/runs/${arm_high}" \
  "${cache_root}/tmp" "${component_cache}"
exec 9>"${root}/runs/run_dual_glance_box_localization_after_queue_20260822.lock"
flock -n 9 || { echo "Glance box-localization experiment already active" >&2; exit 1; }
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for interpolation/anatomy public queue\n' "$(date -Ins)"
  sleep 30
done

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
  sleep 30
done

train_arm() {
  local gpu="$1" arm="$2" box_weight="$3" run_dir="${root}/runs/$2"
  [[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]] && return
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_focus \
    --partial_label_teacher --glance_teacher_labels --hybrid_cohort_glance_labels \
    --complete_label_cohorts_json "${complete_manifest}" \
    --glance_rl_weight 0 --glance_classification_weight 1 \
    --glance_box_localization --glance_box_localization_weight "${box_weight}" \
    --glance_pseudo_positive_threshold 0.95 --glance_pseudo_negative_threshold 0.05 \
    --glance_pseudo_positive_min_voxels 64 --glance_teacher_batch_size 4 \
    --cancer_positive_weight 4 --cancer_negative_weight 1 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json "${empty_healthy}" \
    --healthy_repeat 1 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 283 --val_every 3 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 20000 --workers 1 --glance_lr 3e-5 --warmup_epochs 1 \
    --component_uniform_crop \
    --component_crop_size_power 0.5 --component_crop_points_per_lesion 64 \
    --component_background_points 4096 --component_crop_cache_dir "${component_cache}" \
    --component_legacy_cache_dir "${cache_root}/component_uniform_v2" \
    --enable_scale_intensity_aug --enable_contrast_aug \
    --RandScaleIntensityd_prob 0.10 --RandAdjustContrastd_prob 0.20 \
    --contrast_gamma_min 0.70 --contrast_gamma_max 1.50 \
    --harmonize_cached_chest_window --seed 20260822 \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_low}" 0.05 & pid_low=$!
train_arm 1 "${arm_high}" 0.15 & pid_high=$!
status=0
wait "${pid_low}" || status=$?
wait "${pid_high}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_low}" ARM_B="${arm_high}" EVAL_EPOCHS="283" \
  GLANCE_THRESHOLDS="0.20 0.35 0.50" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_glance_box_localization_public_20260822.sh"
bash "${root}/save_boxloc_candidate_20260822.sh"
