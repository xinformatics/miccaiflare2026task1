#!/usr/bin/env bash
set -euo pipefail

# Public-label diagnostic only. It cannot train, inspect hidden data, package,
# or upload. The predeclared selector decides whether a stronger 0.20/0.80
# Tversky follow-up is scientifically warranted after the 0.25/0.75 branch.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
labels="${project_root}/validation/Validation-Public-Labels"
balanced_root="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826"
source_root="${root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826"
balanced_terminal="${balanced_root}/TERMINAL.json"
source_terminal="${source_root}/TERMINAL.json"
output="${root}/analysis/source_recall_loss_diagnostic_20260826"
terminal="${output}/TERMINAL.json"
selector="${root}/select_recall_loss_escalation.py"

mkdir -p "${output}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "source recall-loss diagnostic already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) exit 0 ;;
    FAILED)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected recall-loss diagnostic terminal: ${status}" >&2; exit 2 ;;
  esac
fi
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    "${python_bin}" - "${line}" "${status}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
line,status,output=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3])
p={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
   "status":"FAILED","decision":"RETRY_PUBLIC_DIAGNOSTIC",
   "reason":f"failed at line {line}, status {status}","public_read":True,
   "hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for prerequisite in "${balanced_terminal}" "${source_terminal}"; do
  while [[ ! -s "${prerequisite}" ]]; do
    printf '%s waiting for fixed-public endpoint terminal: %s\n' "$(date -Ins)" "${prerequisite}"
    sleep 30
  done
done
"${python_bin}" - "${balanced_terminal}" "${source_terminal}" <<'PY'
import json,sys
for path in sys.argv[1:]:
 p=json.load(open(path,encoding="utf-8"))
 if (p.get("status")!="COMPLETE" or p.get("predeclared_epochs")!=[40,80]
     or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
  raise SystemExit(f"invalid fixed-public campaign terminal: {path}")
PY
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

for branch in balanced source; do
  if [[ "${branch}" == balanced ]]; then campaign="${balanced_root}"; else campaign="${source_root}"; fi
  for epoch in 40 80; do
    epoch4=$(printf '%04d' "${epoch}")
    predictions="${campaign}/epoch_${epoch4}/predictions"
    work="${output}/${branch}_epoch_${epoch4}"
    [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
    mkdir -p "${work}"
    "${python_bin}" -u "${gf_root}/analyze_lesion_errors.py" \
      --labels "${labels}" --predictions "${predictions}" \
      --output "${work}/lesion_errors" >"${work}/lesion_errors.log" 2>&1
    "${python_bin}" -u "${gf_root}/analyze_dsc_nsd_gap.py" \
      --labels "${labels}" --predictions "${predictions}" \
      --output "${work}/dsc_nsd_gap" --radius-mm 2 \
      >"${work}/dsc_nsd_gap.log" 2>&1
  done
done

"${python_bin}" -u "${selector}" --balanced-root "${balanced_root}" \
  --source-root "${source_root}" --diagnostic-root "${output}" \
  --output "${terminal}" >"${output}/select.log" 2>&1
trap - EXIT
