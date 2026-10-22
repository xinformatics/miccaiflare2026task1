#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
matched="${root}/public_sweeps/stunetb_matched_e160_20260826"
matched_terminal="${matched}/TERMINAL.json"
analysis="${matched}/component_ensemble.json"
checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_inference_epoch_0160.pth"
public="${root}/public_sweeps/stunetb_component_e160_p70_qualified_20260826"
summary="${public}/summary.tsv"
terminal="${public}/TERMINAL.json"

mkdir -p "${public}"
"${python_bin}" - "${matched_terminal}" "${analysis}" "${checkpoint}" \
  "${summary}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
matched_path,analysis_path,checkpoint,summary_path,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
matched=json.load(open(matched_path,encoding="utf-8"))
analysis=json.load(open(analysis_path,encoding="utf-8"))
if (matched.get("status")!="COMPLETE" or matched.get("evaluation_epoch")!=160
    or matched.get("hidden_read") or matched.get("submission_created") or matched.get("uploaded")):
    raise SystemExit("invalid matched epoch-160 public terminal")
expected=matched.get("component_checkpoint",{})
if (Path(expected.get("path","")).resolve()!=checkpoint.resolve()
    or expected.get("sha256")!=sha(checkpoint)):
    raise SystemExit("component checkpoint identity mismatch")
if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
    or analysis.get("role")!="fixed_public_ensemble_selection_only"
    or analysis.get("hidden_read") or analysis.get("submission_created") or analysis.get("uploaded")):
    raise SystemExit("invalid component public analysis")
selected=analysis.get("best_deployment",{})
if (selected.get("configuration")!="gf_anchor_confirm_either_p70"
    or float(selected.get("Lesion_DSC",0))<0.6154316379500749
    or float(selected.get("Lesion_NSD",0))<0.5104225107769019):
    raise SystemExit("component p70 deployment did not clear the public gate")
summary_path.parent.mkdir(parents=True,exist_ok=True)
with open(summary_path,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=("branch","checkpoint_epoch","configuration","Lesion_DSC","Lesion_NSD"),delimiter="\t")
    writer.writeheader();writer.writerow({"branch":"component_box","checkpoint_epoch":160,
      "configuration":selected["configuration"],"Lesion_DSC":selected["Lesion_DSC"],"Lesion_NSD":selected["Lesion_NSD"]})
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION",
 "reason":"explicit user request to materialize the fixed-public-qualified component-box epoch-160 p70 candidate",
 "predeclared_epochs":[160],"selected_epoch":160,"selected_branch":"component_box",
 "selected_configuration":selected["configuration"],
 "selected_fixed_public_DSC":float(selected["Lesion_DSC"]),
 "selected_fixed_public_NSD":float(selected["Lesion_NSD"]),
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "selected_analysis":{"path":str(analysis_path.resolve()),"sha256":sha(analysis_path)},
 "summary":{"path":str(summary_path.resolve()),"sha256":sha(summary_path)},
 "source_matched_terminal":{"path":str(matched_path.resolve()),"sha256":sha(matched_path)},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

export STUNET_PUBLIC_OUTPUT="${public}"
export STUNET_SAVE_OUTPUT="${root}/public_sweeps/stunetb_component_e160_p70_saved_20260826"
export STUNET_HIDDEN_OUTPUT="${root}/hidden_predictions/stunetb_component_e160_p70_20260826"
export STUNET_ALLOWED_EPOCHS=160
export STUNET_HIDDEN_GPU=0
exec /usr/bin/bash "${root}/save_public_qualified_ensemble_20260825.sh"
