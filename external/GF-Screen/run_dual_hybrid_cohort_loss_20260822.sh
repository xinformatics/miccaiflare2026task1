#!/usr/bin/env bash
set -euo pipefail

# High-upside correction to the rich-label experiment.  The earlier full-label
# arm applied complete-mask loss to every one of the 17,445 cancer cases even
# though many source masks are partial.  This matched pair uses full DiceCE
# only for eight explicitly declared multi-lesion cohorts; all other cancer
# sources keep the proven bounding-box/teacher-safe objective.  The only arm
# difference is the 4x versus 8x sampling prior for complete-mask cohorts.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
complete_manifest="${root}/configs/complete_multilesion_cohorts_20260822.json"
arm04="gf_hybrid_complete04_ep280_lr1e5_5ep_20260822"
arm08="gf_hybrid_complete08_ep280_lr1e5_5ep_20260822"
priority04="${root}/configs/multilesion_source_priority04_20260821.json"
priority08="${root}/configs/multilesion_source_priority08_20260821.json"
deployed_output="${root}/public_sweeps/gf_hybrid_complete04_08_deployed_20260822"
upstream_public_marker="${root}/public_sweeps/gf_rich04_08_full_comp050_deployed_20260821/PUBLIC_ONLY_COMPLETE.txt"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${complete_manifest}" "${priority04}" "${priority08}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm04}" "${root}/runs/${arm08}" \
  "${cache_root}/tmp" "${cache_root}/component_uniform_v2"
exec 9>"${root}/runs/run_dual_hybrid_cohort_loss_20260822.lock"
flock -n 9 || { echo "hybrid cohort experiment is already active" >&2; exit 1; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Memory can fall temporarily while the upstream evaluator postprocesses on
# CPU, then rise for its next checkpoint.  Require its durable completion
# marker before considering instantaneous GPU memory.
while [[ ! -s "${upstream_public_marker}" ]]; do
  printf '%s waiting for upstream public evaluation marker\n' "$(date -Ins)"
  sleep 30
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}

# Existing public evaluation owns both cards.  Wait for real device memory to
# fall below 4 GiB on each card rather than racing a process-name heuristic.
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

# Complete the one predeclared 0.40 follow-up on the best 0.50 checkpoint from
# the upstream grid before committing both GPUs to multi-epoch training.
threshold_marker="${root}/public_sweeps/gf_rich08_cp282_l040_deployed_20260822/PUBLIC_ONLY_COMPLETE.txt"
if [[ ! -s "${threshold_marker}" ]]; then
  bash "${root}/evaluate_rich08_cp282_l040_public_20260822.sh"
fi

train_arm() {
  local gpu="$1" arm="$2" priority="$3"
  local run_dir="${root}/runs/${arm}"
  if [[ -s "${run_dir}/checkpoint_epoch_0285.pt" \
      && -s "${run_dir}/model_final.pt" ]]; then
    echo "${arm}: training already complete"
    return
  fi
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
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm04}" "${priority04}" & pid04=$!
train_arm 1 "${arm08}" "${priority08}" & pid08=$!
status=0
wait "${pid04}" || status=$?
wait "${pid08}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

# Three spaced checkpoints bound the trajectory while avoiding a ten-state
# deployment sweep.  Both 0.40 (recall-favoring) and 0.50 thresholds use the
# exact champion postprocessing and organizer metrics.
ARM_A="${arm04}" ARM_B="${arm08}" EVAL_EPOCHS="281 283 285" \
  FOCUS_THRESHOLDS="0.40 0.50" DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_bbox_lr_escape_public_20260821.sh"
