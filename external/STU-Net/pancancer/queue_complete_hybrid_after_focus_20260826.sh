#!/usr/bin/env bash
set -euo pipefail

# Persistently launch completeness-aware STU-Net after the prioritized
# source-stratified and learned-additive campaigns. The remaining long-Focus
# tail is lower priority because its epoch-303 public endpoint regressed.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
additive="${project_root}/external/GF-Screen/public_sweeps/component_meta_recovery_after_sourcestrat_20260826/TERMINAL.json"
aggressive="${root}/runs/source_aggressive_campaign_20260827/TERMINAL.json"
precision="${project_root}/external/GF-Screen/runs/source_stunet_precision_pipeline_20260827/TERMINAL.json"
controller="${root}/run_component_e160_complete_hybrid_continuation_20260826.sh"
run_terminal="${root}/runs/stunetb_componentbox_e160_completehybrid4x_80ep_20260826/TRAINING_TERMINAL.json"
work="${root}/runs/queue_complete_hybrid_after_focus_20260826"
terminal="${work}/TERMINAL.json"
unit="miccai-stunetb-completehybrid-after-focus-20260826"
label_correction="${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/LABEL_CONTRACT_CORRECTION_20260825.json"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "complete-hybrid queue already active" >&2; exit 1; }

"${python_bin}" - "${label_correction}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
correction,output=map(Path,sys.argv[1:]); p=json.load(open(correction,encoding="utf-8"))
if (p.get("status")!="INVALIDATE_COMPLETE_LABEL_INTERPRETATION"
    or "authoritative complete-label negative mining" not in p.get("prohibited_use",[])):
 raise SystemExit("missing authoritative partial-label correction")
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES",
 "reason":"the claimed complete cohorts use challenge-distributed partial masks; empty regions cannot supervise background",
 "label_contract_correction":{"path":str(correction.resolve()),"sha256":sha(correction)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
exit 0

while [[ ! -s "${additive}" ]]; do
  printf '%s waiting for learned-additive campaign decision\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${additive}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"}
    or not p.get("source_stratified_public") or p.get("hidden_labels_read")
    or p.get("uploaded")):
    raise SystemExit("invalid learned-additive campaign terminal")
PY

while [[ ! -s "${aggressive}" ]]; do
  printf '%s waiting for conditional aggressive-loss campaign decision\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${aggressive}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"} or p.get("hidden_labels_read")
    or p.get("uploaded") or p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}):
    raise SystemExit("invalid conditional aggressive-loss campaign terminal")
PY

while [[ ! -s "${precision}" ]]; do
  printf '%s waiting for source-STU learned precision campaign decision\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${precision}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status") not in {"COMPLETE","SKIPPED"}
    or p.get("decision") not in {"CANDIDATE_SAVED","KEEP_EPOCH160_P70"}
    or p.get("hidden_labels_read") or p.get("uploaded")):
    raise SystemExit("invalid source-STU precision pipeline terminal")
PY

if [[ -s "${run_terminal}" ]]; then
  run_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${run_terminal}")
  [[ "${run_status}" == COMPLETE ]] && exit 0
fi
if ! systemctl --user is-active --quiet "${unit}.service"; then
  systemd-run --user --unit="${unit}" \
    --property=MemoryHigh=48G --property=MemoryMax=64G --property=TasksMax=256 \
    --property=Restart=on-failure --property=RestartSec=60s \
    --property=StartLimitBurst=3 --property=StartLimitIntervalSec=3600s \
    /usr/bin/bash "${controller}"
fi

"${python_bin}" - "${additive}" "${aggressive}" "${precision}" "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
additive,aggressive,precision,controller,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"TRAIN_COMPLETE_HYBRID_BEFORE_REGRESSED_FOCUS_TAIL",
 "learned_additive":{"path":str(additive.resolve()),"sha256":sha(additive)},
 "aggressive_loss_campaign":{"path":str(aggressive.resolve()),"sha256":sha(aggressive)},
 "source_stunet_precision":{"path":str(precision.resolve()),"sha256":sha(precision)},
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-stunetb-completehybrid-after-focus-20260826.service",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
