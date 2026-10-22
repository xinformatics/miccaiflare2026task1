#!/usr/bin/env bash
set -euo pipefail

# Complete union-native folds 2-4 only after both predeclared held-out gates
# pass. Patch OOF and checkpoint-family selection only; no public/hidden work.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/candidate_refiner_union_native_training_20260823"
prep_terminal="${analysis}/PREPARATION_TERMINAL.json"
confirmation_terminal="${analysis}/UNION_NATIVE_CONFIRMATION_TERMINAL.json"
confirmation="${analysis}/UNION_NATIVE_CONFIRMATION_DECISION.json"
work="${analysis}/fivefold"
oof="${work}/five_fold_oof_summary.json"
terminal="${work}/UNION_NATIVE_FIVEFOLD_TERMINAL.json"
transfer_save_terminal="${root}/public_sweeps/candidate_refiner_union_transfer_saved_20260823/UNION_TRANSFER_SAVE_ONLY_TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_union_native_b32_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${work}"
exec 9>"${root}/runs/run_candidate_refiner_union_native_fivefold_20260823.lock"
flock -n 9 || { echo "union-native five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","exit_status":int(sys.argv[2]),"failed_line":sys.argv[3],
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${confirmation_terminal}" ]]; do
  printf '%s waiting for union-native untouched-fold confirmation\n' "$(date -Ins)"
  sleep 30
done
confirmation_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${confirmation_terminal}")
[[ "${confirmation_status}" == COMPLETE ]] || { echo "union-native confirmation failed" >&2; exit 2; }
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${confirmation}")
if [[ "${decision}" != EXPAND_FIVE_FOLD ]]; then
  "${python_bin}" - "${confirmation}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_TRANSFER",
 "reason":"union-native two-fold gate did not pass",
 "confirmation":str(source.resolve()),"confirmation_payload":json.loads(source.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
  exit 0
fi

# Preserve the earliest independently qualified candidate before long-running
# folds 2--4 can claim both GPUs. COMPLETE means a ZIP was audited and saved;
# SKIPPED means the fixed public gate rejected it. Historical/failed markers
# do not release this gate.
while :; do
  if [[ -s "${transfer_save_terminal}" ]]; then
    transfer_status=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
      "${transfer_save_terminal}")
    [[ "${transfer_status}" == COMPLETE || "${transfer_status}" == SKIPPED ]] && break
  fi
  printf '%s waiting for matched-transfer save-or-skip terminal\n' "$(date -Ins)"
  sleep 30
done

prep_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${prep_terminal}")
[[ "${prep_status}" == COMPLETE ]]
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep_terminal}")
[[ -s "${index}" ]]
for fold in 0 1; do
  [[ -s "$(fold_dir "${fold}")/COMPLETE.txt" ]]
done

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
gpu_is_free() {
  local used
  used=$(gpu_memory_mib "$1")
  # GPU 0 has two persistent notebook contexts totaling about 1.15 GiB while
  # retaining more than 47 GiB for training. A 1.5-GiB admission ceiling keeps
  # real training/inference jobs excluded while allowing that harmless
  # baseline to participate in the two-GPU fold schedule.
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1536 ]]
}
wait_any_gpu() {
  while :; do
    free0=0; free1=0
    gpu_is_free 0 && free0=1
    gpu_is_free 1 && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "$(gpu_memory_mib 0)" "$(gpu_memory_mib 1)"
    sleep 30
  done
}
wait_any_gpu

train_fold() {
  local gpu="$1" fold="$2" output
  output="$(fold_dir "${fold}")"; mkdir -p "${output}" "${output}/matplotlib"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" MPLCONFIGDIR="${output}/matplotlib" \
    "${python_bin}" -u "${root}/train_candidate_refiner.py" \
    --index "${index}" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
    --sampling-temperature 0 --seed 20260822 \
    --wandb-project pancancer-ct \
    --wandb-name "candidate_refiner_union_native_b32_fold${fold}_20260823" \
    --resume > "${output}/train.log" 2>&1
}
train_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do train_fold "${gpu}" "${fold}"; done; }
status=0
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  train_sequence 0 2 4 & pid0=$!
  train_sequence 1 3 & pid1=$!
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
else
  if [[ "${free0}" -eq 1 ]]; then
    first_gpu=0; second_gpu=1
  else
    first_gpu=1; second_gpu=0
  fi
  train_sequence "${first_gpu}" 2 4 & first_pid=$!
  second_started=0
  while kill -0 "${first_pid}" 2>/dev/null; do
    if gpu_is_free "${second_gpu}"; then
      train_sequence "${second_gpu}" 3 & second_pid=$!
      second_started=1
      break
    fi
    sleep 30
  done
  wait "${first_pid}" || status=$?
  if [[ "${second_started}" -eq 1 ]]; then
    wait "${second_pid}" || status=$?
  elif [[ "${status}" -eq 0 ]]; then
    train_sequence "${first_gpu}" 3 || status=$?
  fi
fi
[[ "${status}" -eq 0 ]] || exit "${status}"

selection_args=()
for strategy_spec in \
  'composite checkpoint_best.pt validation_predictions_best.npz' \
  'segmentation checkpoint_best_segmentation.pt validation_predictions_best_segmentation.npz' \
  'classification checkpoint_best_classification.pt validation_predictions_best_classification.npz'; do
  read -r strategy checkpoint_name prediction_name <<< "${strategy_spec}"
  summary="${work}/five_fold_oof_summary_${strategy}.json"
  args=()
  for fold in 0 1 2 3 4; do
    output="$(fold_dir "${fold}")"
    args+=(--checkpoint "${output}/${checkpoint_name}")
    args+=(--prediction "${output}/${prediction_name}")
  done
  "${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
    --index "${index}" "${args[@]}" --output "${summary}" \
    > "${work}/summarize_${strategy}.log" 2>&1
  selection_args+=(--candidate "${strategy}=${summary}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_oof_strategy.py" \
  "${selection_args[@]}" --output "${work}/checkpoint_selection.json" \
  --selected-summary "${oof}" > "${work}/checkpoint_selection.log" 2>&1
"${python_bin}" - "${confirmation}" "${oof}" "${work}/checkpoint_selection.json" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
confirmation,oof,selection,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","folds":5,"confirmation":str(confirmation.resolve()),
 "patch_oof":str(oof.resolve()),"checkpoint_selection":str(selection.resolve()),
 "patch_oof_payload":json.loads(oof.read_text()),
 "selection_payload":json.loads(selection.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
