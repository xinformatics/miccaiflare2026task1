#!/usr/bin/env bash
set -euo pipefail

# Convert a qualified completeness-aware public selection into the generic
# label-blind hidden replay contract. Never uploads.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
expected_evaluation_branch="${STUNET_EXPECTED_EVALUATION_BRANCH:-complete_hybrid_e160}"
public_root="${STUNET_PUBLIC_SELECTION_ROOT:-${root}/public_sweeps/stunetb_completehybrid_e40_e80_20260826}"
public_terminal="${public_root}/TERMINAL.json"
compat="${STUNET_EXPORT_CONTRACT_ROOT:-${root}/public_sweeps/stunetb_completehybrid_export_contract_20260826}"
compat_terminal="${compat}/TERMINAL.json"
compat_summary="${compat}/summary.tsv"
compat_analysis="${compat}/selected_analysis.json"
save_output="${STUNET_BRANCH_SAVE_OUTPUT:-${root}/public_sweeps/stunetb_completehybrid_saved_20260826}"
hidden_output="${STUNET_BRANCH_HIDDEN_OUTPUT:-${root}/hidden_predictions/stunetb_completehybrid_20260826}"
model="${STUNET_BRANCH_MODEL:-${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160CompleteHybrid__nnUNetPlans__3d_fullres_STUNetB_PartialBox}"
archive_prefix="${STUNET_BRANCH_ARCHIVE_PREFIX:-gf_ch}"
saver="${root}/save_public_qualified_ensemble_20260825.sh"
runtime_gate_terminal="${STUNET_RUNTIME_GATE_TERMINAL:-}"

mkdir -p "${compat}"
while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for %s public terminal\n' "$(date -Ins)" "${expected_evaluation_branch}"
  sleep 30
done
read -r status decision < <("${python_bin}" - "${public_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p.get("status",""),p.get("decision",""))
PY
)
[[ "${status}" == COMPLETE ]]
if [[ "${decision}" != REVIEW_TRAINED_CANDIDATE ]]; then
  [[ "${decision}" == NO_HIDDEN_ARCHIVE ]]
  printf '%s %s endpoints missed public gates; no hidden inference\n' "$(date -Ins)" "${expected_evaluation_branch}"
  exit 0
fi

# Full-volume jobs on separate GPUs have repeatedly exhausted shared host RAM
# and swap on this machine. A branch wrapper may therefore require the complete
# raw-to-anchor runtime proof to finish before any qualified hidden replay is
# allowed to start. This wait is intentionally after the public decision so a
# non-qualifying branch exits immediately.
if [[ -n "${runtime_gate_terminal}" ]]; then
  while [[ ! -s "${runtime_gate_terminal}" ]]; do
    printf '%s waiting for serialized runtime gate before %s hidden replay\n' \
      "$(date -Ins)" "${expected_evaluation_branch}"
    sleep 30
  done
  "${python_bin}" - "${runtime_gate_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_anchor_equivalence")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid serialized runtime gate")
PY
fi

"${python_bin}" - "${public_terminal}" "${compat_summary}" "${compat_analysis}" \
  "${compat_terminal}" "${model}" "${expected_evaluation_branch}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,summary,analysis,output,model=map(Path,sys.argv[1:6]);expected_branch=sys.argv[6];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();p=json.load(open(source));s=p.get("selected") or {}
epoch=int(s.get("epoch",-1));configuration=s.get("configuration");dsc=float(s.get("Lesion_DSC",-1));nsd=float(s.get("Lesion_NSD",-1))
if (p.get("status")!="COMPLETE" or p.get("decision")!="REVIEW_TRAINED_CANDIDATE"
    or p.get("predeclared_epochs")!=[40,80] or epoch not in (40,80)
    or configuration not in {"stunet_standalone","gf_anchor_confirm_either_p70"}
    or not s.get("public_slot_gate") or dsc<0.6366639257651584 or nsd<0.5347520263176905
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid qualified complete-hybrid selection")
checkpoint=Path(s.get("checkpoint",{}).get("path",""));expected=(model/"fold_all"/f"checkpoint_inference_epoch_{epoch:04d}.pth").resolve()
if checkpoint.resolve()!=expected or s.get("checkpoint",{}).get("sha256")!=sha(checkpoint): raise SystemExit("checkpoint mismatch")
endpoint=Path(s.get("terminal",{}).get("path",""));endpoint_payload=json.load(open(endpoint,encoding="utf-8"))
if (s.get("terminal",{}).get("sha256")!=sha(endpoint)
    or endpoint_payload.get("branch")!=expected_branch
    or endpoint_payload.get("evaluation_epoch")!=epoch):
    raise SystemExit("evaluation branch provenance mismatch")
source_analysis=Path(s.get("analysis",{}).get("path",""))
if s.get("analysis",{}).get("sha256")!=sha(source_analysis): raise SystemExit("source analysis mismatch")
analysis_payload={"schema_version":1,"status":"COMPLETE","case_count":50,
 "role":"fixed_public_ensemble_selection_only","best_deployment":{"configuration":configuration,"Lesion_DSC":dsc,"Lesion_NSD":nsd},
 "source_analysis":{"path":str(source_analysis.resolve()),"sha256":sha(source_analysis)},
 "hidden_read":False,"submission_created":False,"uploaded":False}
temporary=analysis.with_name(f".{analysis.name}.{os.getpid()}.tmp");temporary.write_text(json.dumps(analysis_payload,indent=2)+"\n");os.replace(temporary,analysis)
temporary=summary.with_name(f".{summary.name}.{os.getpid()}.tmp");temporary.write_text("checkpoint_epoch\tconfiguration\tLesion_DSC\tLesion_NSD\tbranch\n"+f"{epoch}\t{configuration}\t{dsc:.17g}\t{nsd:.17g}\tcomponent_box\n");os.replace(temporary,summary)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION",
 "predeclared_epochs":[40,80],"selected_epoch":epoch,"selected_configuration":configuration,"selected_branch":"component_box",
 "selected_fixed_public_DSC":dsc,"selected_fixed_public_NSD":nsd,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "selected_analysis":{"path":str(analysis.resolve()),"sha256":sha(analysis)},"summary":{"path":str(summary.resolve()),"sha256":sha(summary)},
 "source_public_terminal":{"path":str(source.resolve()),"sha256":sha(source)},"public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp");temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY

read -r epoch configuration < <("${python_bin}" - "${compat_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));print(p["selected_epoch"],p["selected_configuration"])
PY
)
short="p70"; [[ "${configuration}" == stunet_standalone ]] && short="solo"
exec env STUNET_PUBLIC_OUTPUT="${compat}" STUNET_ALLOWED_EPOCHS="40,80" \
  STUNET_COMPONENT_MODEL_DIR="${model}" STUNET_SAVE_OUTPUT="${save_output}" \
  STUNET_HIDDEN_OUTPUT="${hidden_output}" STUNET_ARCHIVE_TAG="${archive_prefix}_e${epoch}_${short}_0826" \
  bash "${saver}"
