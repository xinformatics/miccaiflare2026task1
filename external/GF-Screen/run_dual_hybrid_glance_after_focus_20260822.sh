#!/usr/bin/env bash
set -euo pipefail

# Orthogonal Glance experiment queued after the hybrid Focus audit. Both arms
# use the same 4x trusted-complete cohort prior and complete/partial-safe crop
# labels. The only difference is Glance peak LR (3e-5 versus 1e-4).

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/complete_multilesion_cohorts_20260822.json"
priority="${root}/configs/multilesion_source_priority04_20260821.json"
upstream_marker="${root}/public_sweeps/gf_hybrid_complete04_08_deployed_20260822/PUBLIC_ONLY_COMPLETE.txt"
arm_low="gf_glance_hybrid04_lr3e5_ep280_5ep_20260822"
arm_high="gf_glance_hybrid04_lr1e4_ep280_5ep_20260822"
deployed_output="${root}/public_sweeps/gf_glance_hybrid04_lr_deployed_20260822"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${complete_manifest}" "${priority}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_low}" "${root}/runs/${arm_high}" "${cache_root}/tmp"
exec 9>"${root}/runs/run_dual_hybrid_glance_after_focus_20260822.lock"
flock -n 9 || { echo "hybrid Glance experiment is already active" >&2; exit 1; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ ! -s "${upstream_marker}" ]]; do
  printf '%s waiting for hybrid Focus public evaluation marker\n' "$(date -Ins)"
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
  local gpu="$1" arm="$2" lr="$3"
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
    --cancer_positive_weight 1 --cancer_negative_weight 1 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 285 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 1 --glance_lr "${lr}" \
    --warmup_epochs 1 --cohort_priority_json "${priority}" \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_low}" 3e-5 & pid_low=$!
train_arm 1 "${arm_high}" 1e-4 & pid_high=$!
status=0
wait "${pid_low}" || status=$?
wait "${pid_high}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_low}" ARM_B="${arm_high}" EVAL_EPOCHS="281 283 285" \
  GLANCE_THRESHOLDS="0.35 0.50" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_glance_teacher_safe_public_20260821.sh"
