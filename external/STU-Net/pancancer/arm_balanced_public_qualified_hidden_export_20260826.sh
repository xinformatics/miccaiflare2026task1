#!/usr/bin/env bash
set -euo pipefail

# Convert the predeclared balanced epoch-40/80 selection into the rigorously
# validated replay contract used by save_public_qualified_ensemble_20260825.sh.
# This script never uploads. If neither endpoint clears the fixed-public gate,
# it exits without reading hidden images or creating an archive.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
balanced_public="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826"
balanced_terminal="${balanced_public}/TERMINAL.json"
compat_public="${root}/public_sweeps/stunetb_balanced_e40_e80_export_contract_20260826"
compat_terminal="${compat_public}/TERMINAL.json"
compat_summary="${compat_public}/summary.tsv"
save_output="${root}/public_sweeps/stunetb_balanced_e40_e80_saved_20260826"
hidden_output="${root}/hidden_predictions/stunetb_balanced_e40_e80_20260826"
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
saver="${root}/save_public_qualified_ensemble_20260825.sh"

mkdir -p "${compat_public}"
while [[ ! -s "${balanced_terminal}" ]]; do
  printf '%s waiting for balanced epoch-40/80 public terminal\n' "$(date -Ins)"
  sleep 30
done

read -r status decision epoch < <("${python_bin}" - "${balanced_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
selected=p.get("selected") or {}
print(p.get("status",""),p.get("decision",""),selected.get("epoch",""))
PY
)
[[ "${status}" == COMPLETE ]]
if [[ "${decision}" != REVIEW_TRAINED_CANDIDATE ]]; then
  [[ "${decision}" == NO_HIDDEN_ARCHIVE ]]
  printf '%s balanced endpoints did not clear the public slot gate; no hidden inference\n' "$(date -Ins)"
  exit 0
fi
[[ "${epoch}" == 40 || "${epoch}" == 80 ]]
epoch4=$(printf '%04d' "${epoch}")
endpoint_terminal="${balanced_public}/epoch_${epoch4}/TERMINAL.json"

"${python_bin}" - "${balanced_terminal}" "${endpoint_terminal}" "${compat_summary}" \
  "${compat_terminal}" "${model_dir}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path

selection_path,endpoint_path,summary_path,output_path,model_dir=map(Path,sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.load(open(selection_path,encoding="utf-8"))
selected=selection.get("selected") or {}
epoch=int(selected.get("epoch",-1))
if (selection.get("schema_version")!=1 or selection.get("status")!="COMPLETE"
    or selection.get("decision")!="REVIEW_TRAINED_CANDIDATE"
    or selection.get("predeclared_epochs")!=[40,80]
    or epoch not in (40,80) or not selected.get("public_slot_gate")
    or selection.get("hidden_read") or selection.get("submission_created")
    or selection.get("uploaded")):
    raise SystemExit("invalid balanced public selection terminal")
endpoint=json.load(open(endpoint_path,encoding="utf-8"))
metric=endpoint.get("metrics",{}).get("gf_anchor_confirm_either_p70",{})
dsc=float(metric.get("Lesion_DSC",-1)); nsd=float(metric.get("Lesion_NSD",-1))
if (endpoint.get("status")!="COMPLETE" or endpoint.get("evaluation_epoch")!=epoch
    or endpoint.get("branch")!="component_box_e160_balanced"
    or not endpoint.get("public_slot_gate") or dsc!=float(selected["Lesion_DSC"])
    or nsd!=float(selected["Lesion_NSD"])
    or dsc<0.6366639257651584 or nsd<0.5347520263176905
    or endpoint.get("hidden_read") or endpoint.get("submission_created")
    or endpoint.get("uploaded")):
    raise SystemExit("selected balanced endpoint failed identity or metric gate")
checkpoint_item=endpoint.get("checkpoint") or {}
checkpoint=Path(checkpoint_item.get("path",""))
expected=(model_dir/"fold_all"/f"checkpoint_inference_epoch_{epoch:04d}.pth").resolve()
if checkpoint.resolve()!=expected or not checkpoint.is_file() or checkpoint_item.get("sha256")!=sha(checkpoint):
    raise SystemExit("balanced checkpoint identity mismatch")
analysis_item=endpoint.get("analysis") or {}
analysis=Path(analysis_item.get("path",""))
if not analysis.is_file() or analysis_item.get("sha256")!=sha(analysis):
    raise SystemExit("balanced analysis identity mismatch")
analysis_payload=json.load(open(analysis,encoding="utf-8"))
best=analysis_payload.get("best_deployment") or {}
if (analysis_payload.get("role")!="fixed_public_ensemble_selection_only"
    or analysis_payload.get("status")!="COMPLETE"
    or analysis_payload.get("case_count")!=50
    or analysis_payload.get("hidden_read")
    or best.get("configuration")!="gf_anchor_confirm_either_p70"
    or float(best.get("Lesion_DSC",-1))!=dsc or float(best.get("Lesion_NSD",-1))!=nsd):
    raise SystemExit("balanced selected analysis mismatch")
summary=("checkpoint_epoch\tconfiguration\tLesion_DSC\tLesion_NSD\tbranch\n"
         f"{epoch}\tgf_anchor_confirm_either_p70\t{dsc:.17g}\t{nsd:.17g}\tcomponent_box\n")
temporary=summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
temporary.write_text(summary,encoding="utf-8"); os.replace(temporary,summary_path)
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION",
 "predeclared_epochs":[40,80],"selected_epoch":epoch,
 "selected_configuration":"gf_anchor_confirm_either_p70","selected_branch":"component_box",
 "selected_fixed_public_DSC":dsc,"selected_fixed_public_NSD":nsd,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "selected_analysis":{"path":str(analysis.resolve()),"sha256":sha(analysis)},
 "summary":{"path":str(summary_path.resolve()),"sha256":sha(summary_path)},
 "source_balanced_terminal":{"path":str(selection_path.resolve()),"sha256":sha(selection_path)},
 "source_endpoint_terminal":{"path":str(endpoint_path.resolve()),"sha256":sha(endpoint_path)},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output_path)
PY

archive_tag="gf_bal_e${epoch}_p70_0826"
exec env \
  STUNET_PUBLIC_OUTPUT="${compat_public}" \
  STUNET_ALLOWED_EPOCHS="40,80" \
  STUNET_COMPONENT_MODEL_DIR="${model_dir}" \
  STUNET_SAVE_OUTPUT="${save_output}" \
  STUNET_HIDDEN_OUTPUT="${hidden_output}" \
  STUNET_ARCHIVE_TAG="${archive_tag}" \
  bash "${saver}"
