#!/usr/bin/env bash
set -euo pipefail

# Native union fold-0 pilot. It starts only after the shorter prompt-Focus
# pipeline and the fixed transfer baseline have terminated. No public/hidden
# data, packaging, or upload path exists here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/candidate_refiner_union_native_training_20260823"
plan="${analysis}/FUTURE_FOLD_PLAN.json"
prep_terminal="${analysis}/PREPARATION_TERMINAL.json"
baseline_terminal="${analysis}/transfer_baseline_fold0/TRANSFER_BASELINE_TERMINAL.json"
baseline="${analysis}/transfer_baseline_fold0/BASELINE.json"
prompt_terminal="${root}/analysis/focus_prompt_candidate_refiner_pipeline_20260823/PIPELINE_TERMINAL.json"
transfer_saved="${root}/public_sweeps/candidate_refiner_union_transfer_saved_20260823/UNION_TRANSFER_SAVE_ONLY_TERMINAL.json"
output="${root}/runs/candidate_refiner_union_native_b32_fold0_20260823"
decision="${analysis}/UNION_NATIVE_FOLD0_DECISION.json"
terminal="${analysis}/UNION_NATIVE_PILOT_TERMINAL.json"

mkdir -p "${analysis}" "${output}"
exec 9>"${root}/runs/run_candidate_refiner_union_native_pilot_20260823.lock"
flock -n 9 || { echo "union-native pilot already active" >&2; exit 1; }
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

while [[ ! -s "${prep_terminal}" || ! -s "${baseline_terminal}" \
    || ! -s "${prompt_terminal}" || ! -s "${transfer_saved}" ]]; do
  printf '%s waiting for union preparation, prompt pipeline, and matched-transfer preservation\n' "$(date -Ins)"
  sleep 30
done
for completed in "${prep_terminal}" "${baseline_terminal}"; do
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${completed}")
  [[ "${status}" == COMPLETE ]] || { echo "required union audit failed: ${completed}" >&2; exit 2; }
done
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep_terminal}")
[[ -s "${index}" && -s "${baseline}" && -s "${plan}" ]]

gpu="${UNION_NATIVE_GPU:-0}"
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gf_union_native_fold0_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
  --index "${index}" --output "${output}" \
  --folds 5 --fold 0 --epochs 30 --batch-size 8 --workers 2 \
  --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
  --sampling-temperature 0 --seed 20260822 \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_union_native_b32_fold0_20260823 --resume \
  > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_union_native_pilot.py" \
  --plan "${plan}" --baseline "${baseline}" --index "${index}" \
  --checkpoint "composite=${output}/checkpoint_best.pt" \
  --checkpoint "segmentation=${output}/checkpoint_best_segmentation.pt" \
  --checkpoint "classification=${output}/checkpoint_best_classification.pt" \
  --output "${decision}" > "${analysis}/select_union_native_fold0.log" 2>&1
"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
decision,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":str(decision.resolve()),
 "decision_payload":json.loads(decision.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
