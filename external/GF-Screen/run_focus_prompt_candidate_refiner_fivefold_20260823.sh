#!/usr/bin/env bash
set -euo pipefail

# Expand prompt-conditioned Focus only after the fixed fold-0 and untouched
# fold-1 gates pass. Training and patch OOF only; no public/hidden/submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v2_training_20260822/patches_bbox24_min5_v5/index.csv"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
pilot="${root}/analysis/focus_prompt_candidate_refiner_pilot_20260823"
confirmation="${pilot}/CONFIRMATION_DECISION.json"
work="${root}/analysis/focus_prompt_candidate_refiner_fivefold_20260823"
oof="${work}/focus_prompt_patch_oof.json"
terminal="${work}/FOCUS_PROMPT_FIVEFOLD_TERMINAL.json"

fold_dir() {
  printf '%s/runs/focus_prompt_seed_release10_fold%s_20260823' "${root}" "$1"
}

mkdir -p "${work}" "${work}/matplotlib"
exec 9>"${root}/runs/run_focus_prompt_candidate_refiner_fivefold_20260823.lock"
flock -n 9 || { echo "prompt Focus five-fold controller already active" >&2; exit 1; }
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,sys
from pathlib import Path
output=Path(sys.argv[1])
output.write_text(json.dumps({
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","exit_status":int(sys.argv[2]),"failed_line":sys.argv[3],
 "public_read":False,"hidden_read":False,"submission_created":False,
},indent=2)+"\n",encoding="utf-8")
PY
  fi
  exit "${status}"
}
trap write_failure EXIT
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${confirmation}" ]]; do
  printf '%s waiting for prompt Focus untouched-fold confirmation\n' "$(date -Ins)"
  sleep 30
done
decision="$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${confirmation}")"
if [[ "${decision}" != EXPAND_FIVE_FOLD ]]; then
  "${python_bin}" - "${confirmation}" "${terminal}" <<'PY'
import datetime,json,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
output.write_text(json.dumps({
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","reason":"prompt Focus two-fold gate did not pass",
 "confirmation":str(source.resolve()),"confirmation_payload":json.loads(source.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
},indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi

for fold in 0 1; do
  for required in "$(fold_dir "${fold}")/checkpoint_best.pt" \
      "$(fold_dir "${fold}")/validation_predictions_best.npz"; do
    [[ -s "${required}" ]] || { echo "missing confirmed prompt asset: ${required}" >&2; exit 2; }
  done
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0="$(gpu_memory_mib 0)"; used1="$(gpu_memory_mib 1)"
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for both GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

train_fold() {
  local gpu="$1" fold="$2" output
  output="$(fold_dir "${fold}")"
  mkdir -p "${output}" "${output}/matplotlib"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
    "${python_bin}" -u "${root}/train_focus_candidate_refiner.py" \
    --index "${index}" --pretrained "${pretrained}" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 10 --batch-size 8 --workers 2 \
    --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
    --sampling-temperature 0.5 --samples-per-epoch 2048 \
    --freeze-encoder --input-mode soft_seed --seed 20260822 \
    --wandb-project pancancer-ct \
    --wandb-name "focus_prompt_seed_release10_fold${fold}_20260823" \
    --resume > "${output}/train.log" 2>&1
}
train_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do train_fold "${gpu}" "${fold}"; done; }
train_sequence 0 2 4 & pid0=$!
train_sequence 1 3 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

summary_args=()
for fold in 0 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  summary_args+=(--checkpoint "${output}/checkpoint_best.pt")
  summary_args+=(--prediction "${output}/validation_predictions_best.npz")
done
MPLCONFIGDIR="${work}/matplotlib" "${python_bin}" -u \
  "${root}/summarize_focus_candidate_refiner_oof.py" \
  --index "${index}" "${summary_args[@]}" --folds 5 --seed 20260822 \
  --output "${oof}" > "${work}/summarize.log" 2>&1
"${python_bin}" - "${confirmation}" "${oof}" "${terminal}" <<'PY'
import datetime,json,sys
from pathlib import Path
confirmation,oof,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","folds":5,"input_mode":"soft_seed",
 "confirmation":str(confirmation.resolve()),"patch_oof":str(oof.resolve()),
 "patch_oof_payload":json.loads(oof.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
PY

