#!/usr/bin/env bash
set -euo pipefail

# Untouched fold-1 confirmation for union-native learning. It evaluates fixed
# transfer checkpoints first, then trains the native model only after fold 0
# passes. No public/hidden data, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/candidate_refiner_union_native_training_20260823"
plan="${analysis}/FUTURE_FOLD_PLAN.json"
prep_terminal="${analysis}/PREPARATION_TERMINAL.json"
fold0_terminal="${analysis}/UNION_NATIVE_PILOT_TERMINAL.json"
fold0_decision="${analysis}/UNION_NATIVE_FOLD0_DECISION.json"
baseline_root="${analysis}/transfer_baseline_fold1"
baseline="${baseline_root}/BASELINE.json"
transfer_run="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold1_20260823"
output="${root}/runs/candidate_refiner_union_native_b32_fold1_20260823"
decision="${analysis}/UNION_NATIVE_CONFIRMATION_DECISION.json"
terminal="${analysis}/UNION_NATIVE_CONFIRMATION_TERMINAL.json"

mkdir -p "${analysis}" "${baseline_root}" "${output}"
exec 9>"${root}/runs/run_candidate_refiner_union_native_confirmation_20260823.lock"
flock -n 9 || { echo "union-native confirmation already active" >&2; exit 1; }
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

while [[ ! -s "${fold0_terminal}" ]]; do
  printf '%s waiting for union-native fold-0 decision\n' "$(date -Ins)"
  sleep 30
done
fold0_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${fold0_terminal}")
[[ "${fold0_status}" == COMPLETE ]] || { echo "fold-0 union-native pilot failed" >&2; exit 2; }
fold0_result=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${fold0_decision}")
if [[ "${fold0_result}" != CONFIRM_ON_UNTOUCHED_FOLD ]]; then
  "${python_bin}" - "${fold0_decision}" "${decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,decision,terminal=map(Path,sys.argv[1:]); now=datetime.datetime.now().astimezone().isoformat()
payload={
 "schema_version":1,"created_at":now,"decision":"KEEP_TRANSFER","passed":False,
 "reason":"union-native fold-0 gate did not pass","fold0":str(source.resolve()),
 "fold0_payload":json.loads(source.read_text()),"public_read":False,
 "hidden_read":False,"submission_created":False,
}
temporary=decision.with_name(f".{decision.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,decision)
done={"schema_version":1,"created_at":now,"status":"COMPLETE","decision":str(decision.resolve()),
      "decision_payload":payload,"public_read":False,"hidden_read":False,"submission_created":False}
temporary=terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(done,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,terminal)
PY
  exit 0
fi

prep_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${prep_terminal}")
[[ "${prep_status}" == COMPLETE ]]
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep_terminal}")
[[ -s "${index}" && -s "${plan}" ]]
gpu="${UNION_NATIVE_GPU:-0}"
while :; do
  used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

candidate_args=()
for spec in \
  'composite checkpoint_best.pt' \
  'segmentation checkpoint_best_segmentation.pt' \
  'classification checkpoint_best_classification.pt'; do
  read -r family checkpoint_name <<< "${spec}"
  checkpoint="${transfer_run}/${checkpoint_name}"
  result="${baseline_root}/${family}.json"
  predictions="${baseline_root}/${family}.npz"
  [[ -s "${checkpoint}" ]]
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    "${root}/evaluate_candidate_refiner_transfer.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --output "${result}" --predictions "${predictions}" \
    --batch-size 8 --workers 2 --allow-index-transfer \
    > "${baseline_root}/${family}.log" 2>&1
  candidate_args+=(--candidate "${family}=${result}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_transfer_baseline.py" \
  "${candidate_args[@]}" --expected-fold 1 --output "${baseline}" \
  > "${baseline_root}/select.log" 2>&1

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gf_union_native_fold1_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
  --index "${index}" --output "${output}" \
  --folds 5 --fold 1 --epochs 30 --batch-size 8 --workers 2 \
  --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
  --sampling-temperature 0 --seed 20260822 \
  --wandb-project pancancer-ct \
  --wandb-name candidate_refiner_union_native_b32_fold1_20260823 --resume \
  > "${output}/train.log" 2>&1

"${python_bin}" -u "${root}/select_candidate_refiner_union_native_confirmation.py" \
  --plan "${plan}" --fold0 "${fold0_decision}" --baseline "${baseline}" \
  --index "${index}" \
  --checkpoint "composite=${output}/checkpoint_best.pt" \
  --checkpoint "segmentation=${output}/checkpoint_best_segmentation.pt" \
  --checkpoint "classification=${output}/checkpoint_best_classification.pt" \
  --output "${decision}" > "${analysis}/select_union_native_confirmation.log" 2>&1
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
