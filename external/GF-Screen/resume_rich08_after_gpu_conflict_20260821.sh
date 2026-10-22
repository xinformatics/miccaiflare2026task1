#!/usr/bin/env bash
set -euo pipefail

# Exact recovery for the 8x arm after an unrelated val_GF_FLARE process began
# using physical GPU 1 during epoch 282.  Checkpoint 282 contains the completed
# optimizer epoch and scheduler state; resuming it reruns only the final epoch.
# This controller deliberately does not launch public evaluation, which must
# wait until both GPUs are inspected after training.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
arm="gf_rich08_full_comp050_ep280_lr1e5_3ep_20260821"
run_dir="${root}/runs/${arm}"
checkpoint="${run_dir}/checkpoint_epoch_0282.pt"
priority="${root}/configs/multilesion_source_priority08_20260821.json"
marker="${run_dir}/RECOVERY_COMPLETE.txt"

for required in "${checkpoint}" "${priority}" "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing recovery asset: ${required}" >&2; exit 2; }
done
[[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]] && {
  echo "8x final epoch already complete"
  exit 0
}
exec 9>"${run_dir}/recovery.lock"
if ! flock -n 9; then
  echo "8x recovery is already active" >&2
  exit 1
fi

gpu_used_mib() {
  nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}
while [[ "$(gpu_used_mib)" -gt 2048 ]]; do
  printf '%s waiting for physical GPU 1 to become free; used_mib=%s\n' \
    "$(date -Ins)" "$(gpu_used_mib)" | tee -a "${run_dir}/recovery.log"
  sleep 30
done

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_RUN_ID=e5ib6svs
export WANDB_RESUME=allow
cd "${root}"

status=1
for attempt in 1 2 3; do
  printf '%s exact checkpoint-282 recovery attempt %d\n' \
    "$(date -Ins)" "${attempt}" | tee -a "${run_dir}/recovery.log" "${run_dir}/train.log"
  set +e
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --freeze_glance \
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
    >> "${run_dir}/train.log" 2>&1
  status=$?
  set -e
  [[ "${status}" -eq 0 ]] && break
  printf '%s recovery attempt %d exited %d; waiting for GPU 1\n' \
    "$(date -Ins)" "${attempt}" "${status}" | tee -a "${run_dir}/recovery.log"
  while [[ "$(gpu_used_mib)" -gt 2048 ]]; do sleep 30; done
done
[[ "${status}" -eq 0 ]] || exit "${status}"
[[ -s "${run_dir}/checkpoint_epoch_0283.pt" && -s "${run_dir}/model_final.pt" ]]
date -Ins | tee "${marker}"
