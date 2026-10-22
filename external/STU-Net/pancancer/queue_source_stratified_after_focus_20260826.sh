#!/usr/bin/env bash
set -euo pipefail

# Launch the label-safe source/size-stratified STU continuation immediately
# after the balanced path has completed its public/save decision. The remaining
# Focus trajectory is preserved, but runs after this recall-correction branch:
# its epoch-303 fixed-public endpoint already failed to improve the anchor.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
balanced="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826/TERMINAL.json"
balanced_saved="${root}/public_sweeps/stunetb_balanced_e40_e80_saved_20260826/TERMINAL.json"
controller="${root}/run_component_e160_source_stratified_continuation_20260826.sh"
run_terminal="${root}/runs/stunetb_componentbox_e160_sourcesize_f025_cap3_80ep_20260826/TRAINING_TERMINAL.json"
work="${root}/runs/queue_source_stratified_after_focus_20260826"
terminal="${work}/TERMINAL.json"
unit="miccai-stunetb-sourcestrat-after-focus-20260826"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "source-stratified queue already active" >&2; exit 1; }

while [[ ! -s "${balanced}" ]]; do
  printf '%s waiting for balanced epoch-40/80 public decision\n' "$(date -Ins)"
  sleep 30
done
decision=$("${python_bin}" - "${balanced}" <<'PY'
import json,sys
balanced=json.load(open(sys.argv[1],encoding="utf-8"))
if (balanced.get("status")!="COMPLETE" or balanced.get("predeclared_epochs")!=[40,80]
    or balanced.get("hidden_read") or balanced.get("submission_created") or balanced.get("uploaded")):
    raise SystemExit("invalid balanced public terminal")
decision=balanced.get("decision")
if decision not in {"REVIEW_TRAINED_CANDIDATE","NO_HIDDEN_ARCHIVE"}:
    raise SystemExit(f"unexpected balanced decision: {decision}")
print(decision)
PY
)
if [[ "${decision}" == REVIEW_TRAINED_CANDIDATE ]]; then
  while [[ ! -s "${balanced_saved}" ]]; do
    printf '%s waiting for balanced qualified hidden archive terminal\n' "$(date -Ins)"
    sleep 30
  done
  "${python_bin}" - "${balanced_saved}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("decision")!="ARCHIVE_READY_FOR_MANUAL_UPLOAD"
    or not p.get("submission_created") or p.get("uploaded") or p.get("hidden_labels_read")):
    raise SystemExit("balanced hidden-save terminal is not complete and upload-safe")
PY
fi

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

"${python_bin}" - "${decision}" "${balanced}" "${balanced_saved}" "${controller}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
decision=sys.argv[1];balanced,saved,controller,output=map(Path,sys.argv[2:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"QUEUED","decision":"PRIORITIZE_RECALL_CORRECTION_AFTER_BALANCED_EVIDENCE",
 "balanced_decision":decision,
 "balanced_public":{"path":str(balanced.resolve()),"sha256":sha(balanced)},
 "balanced_saved":{"path":str(saved.resolve()),"sha256":sha(saved)} if saved.is_file() else None,
 "training_controller":{"path":str(controller.resolve()),"sha256":sha(controller)},
 "service":"miccai-stunetb-sourcestrat-after-focus-20260826.service",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
