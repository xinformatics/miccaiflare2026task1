#!/usr/bin/env bash
set -euo pipefail

# Complete the checkpointed exact-707 learned additive branch after the
# source-stratified STU campaign. OOF runs before fixed public; hidden inputs
# are touched only by the existing save consumer after both gates pass. No
# controller in this chain uploads a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
source_public="${project}/external/STU-Net/pancancer/public_sweeps/stunetb_sourcestrat_e40_e80_20260826/TERMINAL.json"
source_saved="${project}/external/STU-Net/pancancer/public_sweeps/stunetb_sourcestrat_saved_20260826/TERMINAL.json"
fusion_controller="${root}/run_focus_classifier_fusion_fullvolume_oof_20260824.sh"
fusion_terminal="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824/FOCUS_OOF_TERMINAL.json"
component_controller="${root}/run_component_meta_focus_fusion_fullvolume_oof_20260824.sh"
component_terminal="${root}/analysis/component_meta_focus_fusion_fullvolume_oof_20260824/COMPONENT_META_OOF_TERMINAL.json"
public_controller="${root}/run_component_meta_focus_fusion_public_20260825.sh"
public_terminal="${root}/public_sweeps/component_meta_focus_fusion_20260825/PUBLIC_COMPONENT_TERMINAL.json"
hidden_controller="${root}/save_component_meta_focus_fusion_candidate_20260825.sh"
hidden_terminal="${root}/hidden_predictions/component_meta_focus_fusion_20260825/HIDDEN_COMPONENT_TERMINAL.json"
work="${root}/public_sweeps/component_meta_recovery_after_sourcestrat_20260826"
terminal="${work}/TERMINAL.json"
label_correction="${root}/analysis/fpr_training_mining_707_20260822/LABEL_CONTRACT_CORRECTION_20260825.json"

mkdir -p "${work}" "${root}/runs"
exec 9>"${root}/runs/run_component_meta_recovery_after_sourcestrat_20260826.lock"
flock -n 9 || { echo "component-meta recovery queue already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE || "${status}" == SKIPPED ]] && exit 0
  echo "unexpected additive recovery terminal status: ${status}" >&2; exit 2
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${source_public}" \
    "${source_saved}" "${fusion_terminal}" "${component_terminal}" \
    "${public_terminal}" "${hidden_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
source,saved,fusion,component,public,hidden,output=map(Path,sys.argv[4:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
public_payload=json.load(open(public,encoding="utf-8")) if public.is_file() else {}
hidden_payload=json.load(open(hidden,encoding="utf-8")) if hidden.is_file() else {}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "source_stratified_public":asset(source),"source_stratified_saved":asset(saved),
 "fusion_oof":asset(fusion),"component_oof":asset(component),
 "fixed_public":asset(public),"hidden_save":asset(hidden),
 "public_read":bool(public_payload.get("public_read",False)),
 "hidden_inputs_read":bool(hidden_payload.get("hidden_inputs_read",False)),
 "hidden_labels_read":False,
 "submission_created":bool(hidden_payload.get("submission_created",False)),
 "submission":hidden_payload.get("submission"),
 "uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED RETRY_FROM_VALIDATED_CACHE \
      "component-meta recovery failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

# The challenge contract supersedes the historical upstream-dataset
# interpretation: these 707 masks may omit secondary lesions. This controller
# assigns annotation-negative components to the false-positive class, so it is
# permanently deauthorized rather than merely deprioritized.
"${python_bin}" - "${label_correction}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="INVALIDATE_COMPLETE_LABEL_INTERPRETATION"
    or "authoritative complete-label negative mining" not in p.get("prohibited_use",[])):
 raise SystemExit("missing authoritative 707 label-contract correction")
PY
write_terminal SKIPPED DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES \
  "707 masks may omit secondary lesions; annotation-negative proposals are unknown, not false positives"
trap - EXIT; exit 0

while [[ ! -s "${source_public}" ]]; do
  printf '%s waiting for source-stratified fixed-public decision\n' "$(date -Ins)"
  sleep 30
done
source_decision=$("${python_bin}" - "${source_public}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("predeclared_epochs")!=[40,80]
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
 raise SystemExit("invalid source-stratified public terminal")
decision=p.get("decision")
if decision not in {"REVIEW_TRAINED_CANDIDATE","NO_HIDDEN_ARCHIVE"}:
 raise SystemExit(f"unexpected source-stratified decision: {decision}")
print(decision)
PY
)
if [[ "${source_decision}" == REVIEW_TRAINED_CANDIDATE ]]; then
  while [[ ! -s "${source_saved}" ]]; do
    printf '%s waiting for source-stratified qualified archive\n' "$(date -Ins)"
    sleep 30
  done
  "${python_bin}" - "${source_saved}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("decision")!="ARCHIVE_READY_FOR_MANUAL_UPLOAD"
    or not p.get("submission_created") or p.get("uploaded") or p.get("hidden_labels_read")):
 raise SystemExit("invalid source-stratified save terminal")
PY
fi

for required in "${fusion_controller}" "${component_controller}" \
    "${public_controller}" "${hidden_controller}"; do
  [[ -s "${required}" ]] || { echo "missing additive controller: ${required}" >&2; exit 2; }
done
while :; do
  read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
m={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
p=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
print(m["MemAvailable"],float(next(x.split("=",1)[1] for x in p.split() if x.startswith("avg10="))))
PY
  )
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" <<'PY'
import sys
a,p,g0,g1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if a>=30*1024*1024 and p<=2.0 and g0<=1024 and g1<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe additive launch: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

/usr/bin/bash "${fusion_controller}" frozen >>"${work}/fusion_oof.log" 2>&1
fusion_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${fusion_terminal}")
if [[ "${fusion_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_EXISTING_CHAMPION "fusion OOF ended ${fusion_status}"
  trap - EXIT; exit 0
fi

/usr/bin/bash "${component_controller}" >>"${work}/component_oof.log" 2>&1
/usr/bin/bash "${public_controller}" >>"${work}/fixed_public.log" 2>&1
/usr/bin/bash "${hidden_controller}" >>"${work}/hidden_save.log" 2>&1

public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
hidden_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${hidden_terminal}")
if [[ "${hidden_decision}" == CANDIDATE_SAVED ]]; then
  decision=CANDIDATE_SAVED
elif [[ "${public_decision}" == KEEP_CHAMPION ]]; then
  decision=KEEP_EXISTING_CHAMPION
else
  decision=NO_ADDITIVE_SUBMISSION
fi
write_terminal COMPLETE "${decision}" \
  "completed exact-707 additive OOF, fixed-public gate, and conditional save"
trap - EXIT
