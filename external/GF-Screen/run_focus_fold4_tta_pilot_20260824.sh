#!/usr/bin/env bash
set -euo pipefail

# Local-only x+y flip-TTA pilot on the independently held-out Focus fold 4.
# It waits for the exact ten-epoch trajectory, selects weak/lung epochs using
# no-TTA fold-4 evidence, and tests only those selected checkpoints.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
index="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48/index.csv"
run="${root}/runs/focus_dualwindow_cell707_release10_fold4_20260823"
output="${root}/analysis/focus_dualwindow_cell707_fold4_tta_pilot_20260824"
route_selection="${output}/FOLD4_ROUTE_SELECTION.json"
decision="${output}/DECISION.json"
terminal="${output}/TERMINAL.json"
gpu="${FOCUS_FOLD4_TTA_GPU:-1}"

mkdir -p "${output}/notta" "${output}/tta_xy"
exec 9>"${root}/runs/run_focus_fold4_tta_pilot_20260824.lock"
flock -n 9 || { echo "Focus fold-4 TTA pilot already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_NOTTA","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  fi
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${run}/COMPLETE.txt" ]]; do
  printf '%s waiting for held-out Focus fold 4 to complete\n' "$(date -Ins)"
  sleep 30
done
for epoch in $(seq 0 10); do
  checkpoint="${run}/checkpoint_epoch_$(printf '%04d' "${epoch}").pt"
  prediction="${run}/validation_predictions_epoch_$(printf '%04d' "${epoch}").npz"
  audit="${output}/notta/epoch_$(printf '%04d' "${epoch}").json"
  [[ -s "${checkpoint}" && -s "${prediction}" ]] || {
    echo "missing complete fold-4 epoch ${epoch} artifacts" >&2; exit 2;
  }
  if [[ ! -s "${audit}" ]]; then
    "${python_bin}" "${root}/audit_focus_fold_epoch_sources.py" \
      --index "${index}" --checkpoint "${checkpoint}" \
      --prediction "${prediction}" --folds 5 --fold 4 --seed 20260823 \
      --epoch "${epoch}" --output "${audit}" \
      > "${output}/notta/epoch_$(printf '%04d' "${epoch}").log" 2>&1
  fi
done

route_args=()
for epoch in $(seq 0 10); do
  route_args+=(--audit "${output}/notta/epoch_$(printf '%04d' "${epoch}").json")
done
"${python_bin}" "${root}/select_focus_single_fold_routes.py" \
  "${route_args[@]}" --expected-epochs 11 --output "${route_selection}" \
  > "${output}/select_routes.log" 2>&1

mapfile -t selected_epochs < <("${python_bin}" - "${route_selection}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(*sorted({int(p["selected"][g]["epoch"]) for g in ("weak_sources","lung_cells")}),sep="\n")
PY
)
[[ "${#selected_epochs[@]}" -ge 1 && "${#selected_epochs[@]}" -le 2 ]] || exit 2

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used=$(gpu_memory_mib "${gpu}")
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for TTA pilot GPU %s: %s MiB\n' \
    "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gf_focus_fold4_tta_pilot_20260824"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
tta_audit_args=()
for epoch in "${selected_epochs[@]}"; do
  stem="epoch_$(printf '%04d' "${epoch}")"
  checkpoint="${run}/checkpoint_epoch_$(printf '%04d' "${epoch}").pt"
  prediction="${output}/tta_xy/${stem}.npz"
  summary="${output}/tta_xy/${stem}.summary.json"
  audit="${output}/tta_xy/${stem}.audit.json"
  if [[ ! -s "${summary}" || ! -s "${prediction}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      "${root}/evaluate_focus_checkpoint_tta.py" \
      --index "${index}" --checkpoint "${checkpoint}" \
      --prediction "${prediction}" --summary "${summary}" \
      --folds 5 --fold 4 --seed 20260823 --tta-axes xy \
      --batch-size 8 --workers 2 --segmentation-threshold-max 0.95 \
      > "${output}/tta_xy/${stem}.log" 2>&1
  fi
  "${python_bin}" "${root}/audit_focus_fold_epoch_sources.py" \
    --index "${index}" --checkpoint "${checkpoint}" \
    --prediction "${prediction}" --folds 5 --fold 4 --seed 20260823 \
    --epoch "${epoch}" --output "${audit}" \
    > "${output}/tta_xy/${stem}.audit.log" 2>&1
  tta_audit_args+=(--tta-audit "${audit}")
done

"${python_bin}" "${root}/select_focus_fold_tta_pilot.py" \
  --route-selection "${route_selection}" "${tta_audit_args[@]}" \
  --output "${decision}" > "${output}/select_tta.log" 2>&1
"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.load(source.open())
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "eligible_groups":decision["eligible_groups"],
 "source":str(source.resolve()),
 "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
trap - EXIT
cat "${decision}"
