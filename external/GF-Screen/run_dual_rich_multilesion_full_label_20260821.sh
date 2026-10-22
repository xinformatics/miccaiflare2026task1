#!/usr/bin/env bash
set -euo pipefail

# High-upside recall experiment: learn from cohorts that already contain real
# multi-lesion annotations instead of reinforcing incomplete pseudo-labels.
# Both arms use full-label Focus supervision. The only arm difference is the
# positive crop-centre distribution (legacy voxel-uniform versus sqrt-balanced
# connected components). Public evaluation uses the exact deployed champion
# transform; hidden masks are saved only after a paired PROMOTE decision.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
priority="${root}/configs/multilesion_source_priority16_20260821.json"
arm_a="gf_rich16_full_ep280_lr1e5_3ep_20260821"
arm_b="gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821"
deployed_output="${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" "${priority}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
mkdir -p "${root}/runs/${arm_a}" "${root}/runs/${arm_b}" \
  "${cache_root}/tmp" "${cache_root}/component_uniform_v2"
exec 9>"${root}/runs/run_dual_rich_multilesion_full_label_20260821.lock"
if ! flock -n 9; then
  echo "rich multi-lesion experiment is already active" >&2
  exit 1
fi
export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

train_arm() {
  local gpu="$1" arm="$2" component_mode="$3"
  local run_dir="${root}/runs/${arm}"
  local component_args=()
  if [[ "${component_mode}" == "sqrt" ]]; then
    component_args=(
      --component_uniform_crop --component_crop_size_power 0.5
      --component_crop_points_per_lesion 64
      --component_crop_cache_dir "${cache_root}/component_uniform_v2"
    )
  fi
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
    --cohort_priority_json "${priority}" "${component_args[@]}" \
    --wandb_project pancancer-ct --wandb_name "${arm}" \
    > "${run_dir}/train.log" 2>&1
}

train_arm 0 "${arm_a}" legacy & pid_a=$!
train_arm 1 "${arm_b}" sqrt & pid_b=$!
status=0
wait "${pid_a}" || status=$?
wait "${pid_b}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

ARM_A="${arm_a}" ARM_B="${arm_b}" EVAL_EPOCHS="281 282 283" \
  DEPLOYED_OUTPUT="${deployed_output}" \
  bash "${root}/evaluate_dual_bbox_lr_escape_public_20260821.sh"

# Rank in the actual deployment space. A ZIP is created only if the same
# paired robustness selector used for the champion says PROMOTE.
rank=0
while IFS=$'\t' read -r arm checkpoint_tag candidate_checkpoint focus dsc nsd \
    combined metrics robust; do
  [[ "${arm}" == "variant" ]] && continue
  decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  [[ "${decision}" == "PROMOTE" ]] || continue
  rank=$((rank + 1))
  short_arm=std
  [[ "${arm}" == *comp050* ]] && short_arm=c50
  epoch_tag="${checkpoint_tag#checkpoint_epoch_}"
  focus_tag="${focus/./}"
  tag="gf_rich16_${short_arm}_e${epoch_tag}_l${focus_tag}"
  CHECKPOINT="${candidate_checkpoint}" TAG="${tag}" METRICS="${metrics}" \
    ROBUST_JSON="${robust}" GLANCE_THRESHOLD=0.50 LESION_THRESHOLD="${focus}" \
    bash "${root}/materialize_deployed_candidate_submission_20260821.sh"
  [[ "${rank}" -ge 2 ]] && break
done < "${deployed_output}/summary_by_dsc.tsv"

if [[ "${rank}" -eq 0 ]]; then
  printf '%s\n' 'No rich multi-lesion candidate passed the paired robustness gate; no ZIP saved.' \
    > "${deployed_output}/NO_PROMOTED_SUBMISSION.txt"
fi
