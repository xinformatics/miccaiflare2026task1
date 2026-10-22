#!/usr/bin/env bash
set -euo pipefail

# Sequential exact-707 precision campaign for source-stratified STU proposals.
# It starts only after the higher-priority additive/aggressive campaigns have
# released both GPUs, performs training-only OOF before fixed-public scoring,
# and creates a hidden archive only through the public-qualified save script.
# No controller in this chain can upload to Codabench.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
stunet_root="${project}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
aggressive="${stunet_root}/runs/source_aggressive_campaign_20260827/TERMINAL.json"
proposal_controller="${stunet_root}/run_source_stunet_verified707_proposals_20260827.sh"
proposal_terminal="${stunet_root}/analysis/source_stunet_verified707_proposals_20260827/PROPOSALS_COMPLETE.json"
training_controller="${root}/run_source_stunet_verified707_refiner_20260827.sh"
training_terminal="${root}/analysis/source_stunet_verified707_refiner_20260827/TRAINING_TERMINAL.json"
oof_controller="${root}/run_source_stunet_verified707_fullvolume_oof_20260827.sh"
oof_terminal="${root}/analysis/source_stunet_verified707_fullvolume_oof_20260827/FULL_VOLUME_OOF_TERMINAL.json"
public_controller="${root}/run_source_stunet_refiner_public_save_20260827.sh"
public_terminal="${root}/public_sweeps/source_stunet_refiner_20260827/TERMINAL.json"
work="${root}/runs/source_stunet_precision_pipeline_20260827"
terminal="${work}/TERMINAL.json"
label_correction="${root}/analysis/fpr_training_mining_707_20260822/LABEL_CONTRACT_CORRECTION_20260825.json"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "source-STU precision pipeline already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE || "${status}" == SKIPPED ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${aggressive}" \
    "${proposal_terminal}" "${training_terminal}" "${oof_terminal}" \
    "${public_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]; paths=list(map(Path,sys.argv[4:-1])); output=Path(sys.argv[-1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
public=json.load(open(paths[4],encoding="utf-8")) if paths[4].is_file() else {}
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "aggressive_campaign":asset(paths[0]),"proposals":asset(paths[1]),
 "five_fold_training":asset(paths[2]),"full_volume_oof":asset(paths[3]),
 "fixed_public_and_save":asset(paths[4]),
 "public_read":bool(public.get("public_read",False)),
 "hidden_inputs_read":bool(public.get("hidden_inputs_read",False)),
 "hidden_labels_read":False,"submission_created":bool(public.get("submission_created",False)),
 "submission":public.get("submission"),"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED RETRY_FROM_VALIDATED_STAGE "pipeline failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

# This implementation needs proposal negatives for a precision classifier.
# The 707 challenge-distributed masks are partial under info.txt, even when
# their upstream source datasets had exhaustive annotations. Never launch the
# negative-mining stages under the invalidated complete-label interpretation.
"${python_bin}" - "${label_correction}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="INVALIDATE_COMPLETE_LABEL_INTERPRETATION"
    or "authoritative complete-label negative mining" not in p.get("prohibited_use",[])):
 raise SystemExit("missing authoritative 707 label-contract correction")
PY
write_terminal SKIPPED DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES \
  "source-STU precision classifier requires negatives unavailable under the partial-label contract"
trap - EXIT; exit 0

while [[ ! -s "${aggressive}" ]]; do
  printf '%s waiting for conditional aggressive-loss campaign\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${aggressive}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"} or p.get("hidden_labels_read")
    or p.get("uploaded") or p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}):
 raise SystemExit("invalid aggressive-loss prerequisite")
PY

for controller in "${proposal_controller}" "${training_controller}" \
    "${oof_controller}" "${public_controller}"; do
  [[ -s "${controller}" ]] || { echo "missing source-STU pipeline controller: ${controller}" >&2; exit 2; }
done

/usr/bin/bash "${proposal_controller}" >>"${work}/proposals.log" 2>&1
/usr/bin/bash "${training_controller}" >>"${work}/five_fold_training.log" 2>&1
/usr/bin/bash "${oof_controller}" >>"${work}/full_volume_oof.log" 2>&1
/usr/bin/bash "${public_controller}" >>"${work}/fixed_public_and_save.log" 2>&1

read -r final_status final_decision < <("${python_bin}" - "${public_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"} or p.get("hidden_labels_read")
    or p.get("uploaded")):
 raise SystemExit("invalid source-STU public/save terminal")
print(p["status"],p["decision"])
PY
)
if [[ "${final_decision}" == CANDIDATE_SAVED ]]; then
  decision=CANDIDATE_SAVED
elif [[ "${final_decision}" == KEEP_EPOCH160_P70 ]]; then
  decision=KEEP_EPOCH160_P70
else
  echo "unexpected source-STU terminal decision: ${final_decision}" >&2
  exit 2
fi
write_terminal "${final_status}" "${decision}" \
  "completed exact-707 source-STU proposal confirmation, full-volume OOF, and fixed-public gate"
trap - EXIT
