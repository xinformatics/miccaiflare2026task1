#!/usr/bin/env bash
set -euo pipefail

# Fill GPU 1 after the primary fold-3 sequence releases it. These are the exact
# two fold-0 training jobs consumed by the gated Focus pilot; this controller
# performs no selection, public/hidden inference, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
fold3_complete="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold3_20260823/COMPLETE.txt"
pilot_decision="${root}/analysis/focus_candidate_refiner_pilot_20260823/PILOT_DECISION.json"

exec 9>"${root}/runs/pretrain_focus_ab_after_fold3_20260823.lock"
flock -n 9 || { echo "early Focus A/B controller already active" >&2; exit 1; }
while [[ ! -s "${fold3_complete}" ]]; do
  printf '%s waiting for primary fold 3 completion\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${pilot_decision}" ]] && exit 0
for required in "${index}" "${pretrained}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
while :; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for released GPU 1: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done

batch_size=$(CUDA_VISIBLE_DEVICES=1 bash "${root}/ensure_focus_batch_size_20260823.sh")

train_variant() {
  local recipe="$1" output="$2"
  mkdir -p "${output}" "${output}/matplotlib"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR="${output}/matplotlib" \
    flock "${output}/train.lock" \
    "${python_bin}" -u "${root}/train_focus_candidate_refiner.py" \
    --index "${index}" --pretrained "${pretrained}" --output "${output}" \
    --folds 5 --fold 0 --epochs 3 --batch-size "${batch_size}" --workers 2 \
    --lr 3e-5 --weight-decay 1e-5 --loss-recipe "${recipe}" \
    --sampling-temperature 0.5 --samples-per-epoch 2048 \
    --freeze-encoder --seed 20260822 --wandb-project pancancer-ct \
    --wandb-name "focus_candidate_refiner_${recipe}_fold0_20260823" \
    --resume > "${output}/train.log" 2>&1
}

train_variant release_dice_ce \
  "${root}/runs/focus_candidate_refiner_release_fold0_20260823"
train_variant boundary_balanced \
  "${root}/runs/focus_candidate_refiner_boundary_fold0_20260823"
date -Ins > "${root}/runs/FOCUS_AB_EARLY_COMPLETE.txt"
