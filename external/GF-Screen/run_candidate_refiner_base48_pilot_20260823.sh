#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
output="${root}/runs/candidate_refiner_base48_fold0_20260823"
plan="${root}/analysis/candidate_refiner_base48_pilot_20260823/PLAN.json"
expected_hash="b1b077a1bf37647f9e6000262d8f0acf1564fa030d3054094e8e6176dd92e5b1"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_base48_pilot_20260823.lock"
flock -n 9 || { echo "base48 pilot already active" >&2; exit 1; }
[[ -s "${output}/COMPLETE.txt" ]] && exit 0
[[ -s "${plan}" ]] || { echo "missing predeclared plan" >&2; exit 1; }
observed_hash="$(sha256sum "${index}" | awk '{print $1}')"
[[ "${observed_hash}" == "${expected_hash}" ]] || {
  echo "index hash mismatch: ${observed_hash}" >&2
  exit 1
}

while :; do
  used="$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gfref_base48_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/train_candidate_refiner.py" \
  --index "${index}" --output "${output}" \
  --folds 5 --fold 0 --epochs 12 --batch-size 8 --workers 2 \
  --lr 3e-4 --weight-decay 1e-4 --base-channels 48 \
  --sampling-temperature 0.0 --seed 20260822 \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_base48_casebalanced_fold0_12ep \
  --resume

