#!/usr/bin/env bash
set -euo pipefail

# Conditional campaign: wait for the predeclared loss diagnostic and the
# already-running additive campaign, then train/evaluate/save alpha=.20,
# beta=.80 only if qualified. Creates no upload and reads hidden images only
# through the existing public-qualified save contract.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
diagnostic="${root}/analysis/source_recall_loss_diagnostic_20260826/TERMINAL.json"
additive="${project_root}/external/GF-Screen/public_sweeps/component_meta_recovery_after_sourcestrat_20260826/TERMINAL.json"
trainer="${root}/run_component_e160_source_aggressive_continuation_20260827.sh"
training_terminal="${root}/runs/stunetb_componentbox_e160_source_a020b080_80ep_20260827/TRAINING_TERMINAL.json"
evaluator="${root}/run_source_aggressive_e40_e80_public_20260827.sh"
public_terminal="${root}/public_sweeps/stunetb_sourceaggressive_e40_e80_20260827/TERMINAL.json"
saver="${root}/arm_source_aggressive_qualified_hidden_20260827.sh"
save_terminal="${root}/public_sweeps/stunetb_sourceaggressive_saved_20260827/TERMINAL.json"
runtime_gate_terminal="${project_root}/external/GF-Screen/docker_submission/analysis/runtime_anchor_equivalence_20260827/EQUIVALENCE.json"
work="${root}/runs/source_aggressive_campaign_20260827"
terminal="${work}/TERMINAL.json"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "source-aggressive campaign already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE || "${status}" == SKIPPED ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${diagnostic}" \
    "${additive}" "${training_terminal}" "${public_terminal}" "${save_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]; paths=list(map(Path,sys.argv[4:-1])); output=Path(sys.argv[-1])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
public=json.load(open(paths[3],encoding="utf-8")) if paths[3].is_file() else {}
saved=json.load(open(paths[4],encoding="utf-8")) if paths[4].is_file() else {}
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,"loss":{"alpha_fp":0.20,"beta_fn":0.80},
 "diagnostic":asset(paths[0]),"prior_additive_campaign":asset(paths[1]),
 "training":asset(paths[2]),"fixed_public":asset(paths[3]),"hidden_save":asset(paths[4]),
 "public_read":bool(public.get("public_read",False)),
 "hidden_inputs_read":bool(saved.get("hidden_inputs_read",False)),"hidden_labels_read":False,
 "submission_created":bool(saved.get("submission_created",False)),
 "submission":saved.get("submission"),"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED RETRY_CONDITIONAL_CAMPAIGN "failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

while [[ ! -s "${diagnostic}" ]]; do
  printf '%s waiting for source recall-loss diagnostic\n' "$(date -Ins)"
  sleep 30
done
decision=$("${python_bin}" - "${diagnostic}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("decision") not in
    {"QUALIFY_TVERSKY_A020_B080_FOLLOWUP","KEEP_A025_B075_NO_STRONGER_LOSS"}
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid loss diagnostic terminal")
print(p["decision"])
PY
)
if [[ "${decision}" == KEEP_A025_B075_NO_STRONGER_LOSS ]]; then
  write_terminal SKIPPED KEEP_A025_B075 "moderate Tversky did not justify stronger FN pressure"
  trap - EXIT; exit 0
fi

while [[ ! -s "${additive}" ]]; do
  printf '%s waiting for prioritized additive campaign before aggressive training\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${additive}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"} or p.get("hidden_labels_read") or p.get("uploaded")):
    raise SystemExit("invalid additive campaign terminal")
PY

for controller in "${trainer}" "${evaluator}" "${saver}"; do
  [[ -s "${controller}" ]] || { echo "missing aggressive campaign controller: ${controller}" >&2; exit 2; }
done

# The public terminal also releases several full-volume deployment proofs and
# a qualified hidden saver. Keep this two-GPU training campaign behind the
# complete runtime chain so independent controllers cannot contend for host
# memory while swap is exhausted.
while [[ ! -s "${runtime_gate_terminal}" ]]; do
  printf '%s waiting for serialized deployment runtime gate before aggressive training\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${runtime_gate_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_anchor_equivalence")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid serialized deployment runtime gate")
PY
/usr/bin/bash "${trainer}" >>"${work}/train.log" 2>&1
/usr/bin/bash "${evaluator}" >>"${work}/fixed_public.log" 2>&1
public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" == REVIEW_TRAINED_CANDIDATE ]]; then
  /usr/bin/bash "${saver}" >>"${work}/hidden_save.log" 2>&1
  [[ -s "${save_terminal}" ]]
  write_terminal COMPLETE CANDIDATE_SAVED "aggressive loss passed fixed-public gates; archive is ready for manual review"
else
  [[ "${public_decision}" == NO_HIDDEN_ARCHIVE ]]
  write_terminal COMPLETE KEEP_BEST_EXISTING "aggressive loss missed the fixed-public DSC/NSD gates"
fi
trap - EXIT
