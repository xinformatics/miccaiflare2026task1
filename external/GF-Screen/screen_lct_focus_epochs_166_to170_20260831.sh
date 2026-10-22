#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run_dir="${root}/runs/gf_lct947_focus_e160_to170_20260830"
work="${root}/analysis/lct_task2_finetune_20260830"
lock="${root}/runs/screen_lct_focus_epochs_166_to170_20260831.lock"

mkdir -p "${work}"
exec 9>"${lock}"
flock -n 9 || { echo "Focus checkpoint screener already active" >&2; exit 1; }
export MPLCONFIGDIR=/tmp/miccai26_mpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for epoch in 166 167 168 169 170; do
  checkpoint="${run_dir}/checkpoint_epoch_0${epoch}.pt"
  output="${work}/focus_component_screen_ep${epoch}.json"
  [[ -s "${output}" ]] && continue
  while [[ ! -s "${checkpoint}" ]]; do sleep 30; done
  while :; do
    used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits \
      | tr -d '[:space:]')
    if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 2048 ]]; then break; fi
    sleep 30
  done
  cd "${root}"
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u evaluate_focus_component_crops.py \
    --checkpoint "ep${epoch}=${checkpoint}" --output "${output}" \
    --gpu 0 --batch-size 1 --cases-per-group 8 --components-per-case 3
done
