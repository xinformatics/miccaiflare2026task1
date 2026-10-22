#!/usr/bin/env bash
set -euo pipefail

# Prepared controlled experiment. Launch only after primary qualification and
# save-only packaging have released a GPU; this script itself cannot read
# public/hidden data or create a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
output="${root}/runs/focus_prompt_seed_release10_fold0_20260823"
plan="${root}/analysis/focus_prompt_candidate_refiner_pilot_20260823/PLAN.json"
expected_index="b1b077a1bf37647f9e6000262d8f0acf1564fa030d3054094e8e6176dd92e5b1"
expected_pretrained="b60d1c279a574ce9c2a183a12a28e03388d7a8001ab2eeffdb1198460e1d19be"

mkdir -p "${output}"
exec 9>"${root}/runs/run_focus_prompt_candidate_refiner_pilot_20260823.lock"
flock -n 9 || { echo "prompt Focus pilot already active" >&2; exit 1; }
[[ -s "${output}/COMPLETE.txt" ]] && exit 0
[[ -s "${plan}" ]] || { echo "missing predeclared plan" >&2; exit 2; }
[[ "$(sha256sum "${index}" | awk '{print $1}')" == "${expected_index}" ]] || {
  echo "index hash mismatch" >&2; exit 2;
}
[[ "$(sha256sum "${pretrained}" | awk '{print $1}')" == "${expected_pretrained}" ]] || {
  echo "pretrained hash mismatch" >&2; exit 2;
}

gpu="${FOCUS_PROMPT_GPU:-1}"
while :; do
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gf_focus_prompt_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  "${root}/train_focus_candidate_refiner.py" \
  --index "${index}" --pretrained "${pretrained}" --output "${output}" \
  --folds 5 --fold 0 --epochs 10 --batch-size 8 --workers 2 \
  --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
  --sampling-temperature 0.5 --samples-per-epoch 2048 \
  --freeze-encoder --input-mode soft_seed --seed 20260822 \
  --wandb-project pancancer-ct \
  --wandb-name focus_prompt_seed_release10_fold0_20260823 --resume

"${python_bin}" -u "${root}/select_focus_prompt_pilot.py" \
  --plan "${plan}" --checkpoint "${output}/checkpoint_best.pt" \
  --output "${root}/analysis/focus_prompt_candidate_refiner_pilot_20260823/DECISION.json"
