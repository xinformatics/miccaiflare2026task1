#!/usr/bin/env bash
set -euo pipefail

# Untouched-fold confirmation only. This controller has no public/hidden,
# packaging, or upload path and starts only after a passing fold-0 decision.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/focus_prompt_candidate_refiner_pilot_20260823"
plan="${analysis}/PLAN.json"
fold0="${root}/runs/focus_prompt_seed_release10_fold0_20260823"
fold1="${root}/runs/focus_prompt_seed_release10_fold1_20260823"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"

mkdir -p "${fold1}"
exec 9>"${root}/runs/run_focus_prompt_candidate_refiner_fold1_20260823.lock"
flock -n 9 || { echo "prompt Focus fold-1 confirmation already active" >&2; exit 1; }
[[ -s "${analysis}/CONFIRMATION_DECISION.json" ]] && exit 0
pilot_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["decision"])' "${analysis}/DECISION.json")
if [[ "${pilot_decision}" != CONFIRM_ON_UNTOUCHED_FOLD ]]; then
  "${python_bin}" - "${analysis}/DECISION.json" "${analysis}/CONFIRMATION_DECISION.json" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "decision":"KEEP_BASE32","passed":False,
 "reason":"fold-0 prompt Focus gate did not pass",
 "pilot":str(source.resolve()),"pilot_payload":json.loads(source.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi

gpu="${FOCUS_PROMPT_GPU:-1}"
while :; do
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${fold1}/matplotlib"
export TMPDIR="/tmp/gf_focus_prompt_fold1_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
  "${root}/train_focus_candidate_refiner.py" \
  --index "${index}" --pretrained "${pretrained}" --output "${fold1}" \
  --folds 5 --fold 1 --epochs 10 --batch-size 8 --workers 2 \
  --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
  --sampling-temperature 0.5 --samples-per-epoch 2048 \
  --freeze-encoder --input-mode soft_seed --seed 20260822 \
  --wandb-project pancancer-ct \
  --wandb-name focus_prompt_seed_release10_fold1_20260823 --resume

"${python_bin}" -u "${root}/select_focus_prompt_confirmation.py" \
  --plan "${plan}" --index "${index}" \
  --focus-run "${fold0}" --focus-run "${fold1}" \
  --small-prediction "${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold0_20260823/validation_predictions_best_segmentation.npz" \
  --small-prediction "${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold1_20260823/validation_predictions_best_segmentation.npz" \
  --output "${analysis}/CONFIRMATION_DECISION.json"
