#!/usr/bin/env bash
set -euo pipefail

# Matched follow-up to the 16x rich-multilesion experiment.  The 16x arm made
# 79.8% of the lesion stream come from eight high-multicomponent cohorts and
# improved deployed public metrics after one epoch, but later checkpoints
# regressed.  These two arms test whether gentler 4x/8x sampling retains that
# signal with less distribution shift.  Everything except the cohort prior is
# identical: immutable epoch-280 checkpoint, sqrt component-balanced crops,
# full-label loss, optimizer, schedule, seed, epoch budget, and evaluation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
arm04="gf_rich04_full_comp050_ep280_lr1e5_3ep_20260821"
arm08="gf_rich08_full_comp050_ep280_lr1e5_3ep_20260821"
priority04="${root}/configs/multilesion_source_priority04_20260821.json"
priority08="${root}/configs/multilesion_source_priority08_20260821.json"
deployed_output="${root}/public_sweeps/gf_rich04_08_full_comp050_deployed_20260821"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${priority04}" "${priority08}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm04}" "${root}/runs/${arm08}" \
  "${cache_root}/tmp" "${cache_root}/component_uniform_v2"
exec 9>"${root}/runs/run_dual_rich_factor04_08_20260821.lock"
if ! flock -n 9; then
  echo "4x/8x rich-cohort experiment is already active" >&2
  exit 1
fi
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_arm() {
  local gpu="$1" arm="$2" priority="$3"
  local run_dir="${root}/runs/${arm}"
  if [[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]]; then
    echo "${arm}: training already complete"
    return
  fi
  cd "${root}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 280 \
    --reset_optimizer --reset_best_acc --freeze_glance \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 283 --val_every 3 --batch_size 8 --sw_batch_size 2 \
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

# Public-only evaluation: this helper cannot infer hidden cases or build ZIPs.
ARM_A="${arm04}" ARM_B="${arm08}" EVAL_EPOCHS="281 282 283" \
  DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_bbox_lr_escape_public_20260821.sh"
