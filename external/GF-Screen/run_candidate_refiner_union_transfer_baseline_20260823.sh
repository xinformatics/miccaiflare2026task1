#!/usr/bin/env bash
set -euo pipefail

# Evaluate the three fixed epoch105-trained fold-0 checkpoint families on the
# newly prepared union-native fold-0 patches. No optimization is performed.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analysis="${root}/analysis/candidate_refiner_union_native_training_20260823"
prep_terminal="${analysis}/PREPARATION_TERMINAL.json"
output="${analysis}/transfer_baseline_fold0"
terminal="${output}/TRANSFER_BASELINE_TERMINAL.json"
run="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold0_20260823"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_union_transfer_baseline_20260823.lock"
flock -n 9 || { echo "union transfer baseline already active" >&2; exit 1; }
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

while [[ ! -s "${prep_terminal}" ]]; do
  printf '%s waiting for union-native patch preparation\n' "$(date -Ins)"
  sleep 30
done
prep_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${prep_terminal}")
[[ "${prep_status}" == COMPLETE ]] || { echo "union-native preparation failed" >&2; exit 2; }
index=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["index"])' "${prep_terminal}")
[[ -s "${index}" ]]

gpu="${UNION_TRANSFER_GPU:-0}"
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
  checkpoint="${run}/${checkpoint_name}"
  result="${output}/${family}.json"
  predictions="${output}/${family}.npz"
  [[ -s "${checkpoint}" ]]
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    "${root}/evaluate_candidate_refiner_transfer.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --output "${result}" --predictions "${predictions}" \
    --batch-size 8 --workers 2 --allow-index-transfer \
    > "${output}/${family}.log" 2>&1
  candidate_args+=(--candidate "${family}=${result}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_transfer_baseline.py" \
  "${candidate_args[@]}" --output "${output}/BASELINE.json" \
  > "${output}/select.log" 2>&1
"${python_bin}" - "${output}/BASELINE.json" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
baseline,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","baseline":str(baseline.resolve()),
 "baseline_payload":json.loads(baseline.read_text()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
