#!/usr/bin/env bash
set -euo pipefail

# Sequential fixed-public evaluation for completeness-aware epochs 40 and 80.
# This script cannot inspect hidden data or create/upload a submission.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${root}/predict_stunet_safe.py"
analyzer="${root}/analyze_stunet_gf_ensemble.py"
evaluation_branch="${STUNET_EVALUATION_BRANCH:-complete_hybrid_e160}"
expected_trainer="${STUNET_EXPECTED_TRAINER:-STUNetTrainerPanCancerComponentBoxE160CompleteHybrid}"
model="${STUNET_EVALUATION_MODEL:-${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160CompleteHybrid__nnUNetPlans__3d_fullres_STUNetB_PartialBox}"
fold="${model}/fold_all"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
training_terminal="${STUNET_TRAINING_TERMINAL:-${root}/runs/stunetb_componentbox_e160_completehybrid4x_80ep_20260826/TRAINING_TERMINAL.json}"
output_root="${STUNET_PUBLIC_OUTPUT_ROOT:-${root}/public_sweeps/stunetb_completehybrid_e40_e80_20260826}"
terminal="${output_root}/TERMINAL.json"
epochs=(40 80)

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
mkdir -p "${output_root}"
exec 9>"${output_root}/controller.lock"
flock -n 9 || { echo "${evaluation_branch} public controller already active" >&2; exit 1; }

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" "${output_root}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2]);root=Path(sys.argv[3])
public_read=any(root.rglob("closed_metrics.log"))
payload={"schema_version":1,
 "created_at":datetime.datetime.now().astimezone().isoformat(),"status":"FAILED",
 "reason":reason,"decision":"RETRY_COMPLETE_HYBRID_PUBLIC",
 "public_read":public_read,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "complete-hybrid public controller failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output_root}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected ${evaluation_branch} public terminal: ${existing}" >&2; exit 2 ;;
  esac
fi

while [[ ! -s "${training_terminal}" ]]; do
  printf '%s waiting for %s training terminal\n' "$(date -Ins)" "${evaluation_branch}"
  sleep 30
done
"${python_bin}" - "${training_terminal}" "${evaluation_branch}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
branch=sys.argv[2]
if (p.get("status")!="COMPLETE" or p.get("public_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid STU branch training terminal")
recipe=p.get("recipe",{})
if recipe.get("parent_epoch")!=160 or recipe.get("epochs")!=80 or recipe.get("physical_gpus")!=[0,1]:
    raise SystemExit("STU branch training recipe mismatch")
if branch=="complete_hybrid_e160":
    valid=recipe.get("verified_complete_cases")==3145
elif branch=="source_stratified_e160":
    valid=(recipe.get("case_count")==17575 and recipe.get("source_count")==28
           and recipe.get("case_sampling")=="75% uniform plus 25% size-bin-preserving bounded source target"
           and "outside boxes ignored for every case" in recipe.get("loss",""))
elif branch=="source_aggressive_e160":
    valid=(recipe.get("case_count")==17575 and recipe.get("source_count")==28
           and recipe.get("case_sampling")=="75% uniform plus 25% size-bin-preserving bounded source target"
           and "Tversky a020 b080" in recipe.get("loss","")
           and recipe.get("causal_difference_from_source_a025b075")=="Tversky alpha/beta only")
else:
    raise SystemExit("unsupported STU evaluation branch")
if not valid: raise SystemExit("STU branch-specific training recipe mismatch")
PY

evaluate_epoch() {
  local epoch="$1" epoch4 output predictions analysis checkpoint sidecar checkpoint_sha part_id part4 case_terminal used
  epoch4=$(printf '%04d' "${epoch}")
  output="${output_root}/epoch_${epoch4}"
  predictions="${output}/predictions"
  analysis="${output}/closed_metrics.json"
  checkpoint="${fold}/checkpoint_inference_epoch_${epoch4}.pth"
  sidecar="${fold}/checkpoint_inference_epoch_${epoch4}.json"
  mkdir -p "${output}" "${predictions}"
  [[ -s "${checkpoint}" && -s "${sidecar}" ]]
  "${python_bin}" - "${epoch}" "${checkpoint}" "${sidecar}" "${expected_trainer}" <<'PY'
import json,sys
from pathlib import Path
epoch=int(sys.argv[1]);checkpoint=Path(sys.argv[2]);sidecar=Path(sys.argv[3]);expected=sys.argv[4];p=json.load(open(sidecar))
if (p.get("schema_version")!=1 or p.get("current_epoch")!=epoch
    or p.get("checkpoint_kind")!="inference_only"
    or p.get("checkpoint")!=checkpoint.name or p.get("checkpoint_bytes")!=checkpoint.stat().st_size
    or p.get("trainer_name")!=expected):
    raise SystemExit("invalid STU branch inference sidecar")
PY
  checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
  while :; do
    used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && break
    sleep 10
  done
  for part_id in $(seq 0 49); do
    part4=$(printf '%04d' "${part_id}")
    case_terminal="${output}/HYBRID_E${epoch4}_CASE_${part4}.json"
    if [[ -s "${case_terminal}" ]] && "${python_bin}" - "${case_terminal}" "${predictions}" "${checkpoint_sha}" "${part_id}" <<'PY'
import json,sys
from pathlib import Path
terminal=Path(sys.argv[1]);predictions=Path(sys.argv[2]);sha=sys.argv[3];part=int(sys.argv[4]);p=json.load(open(terminal));cases=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("dataset_role")!="public" or p.get("part_id")!=part
    or p.get("num_parts")!=50 or p.get("case_count")!=1 or len(cases)!=1
    or p.get("checkpoint",{}).get("sha256")!=sha or p.get("public_labels_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
name=cases[0]
if not (predictions/name).is_file() or not (predictions/f"{name.removesuffix('.nii.gz')}.prob16.npz").is_file():
    raise SystemExit(1)
PY
    then continue; fi
    "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
      --model-dir "${model}" --checkpoint "$(basename "${checkpoint}")" --gpu 1 \
      --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities \
      --sequential --dataset-role public --terminal "${case_terminal}" \
      >"${output}/case_${part4}.log" 2>&1
  done
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  [[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
  [[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]
  "${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
    --stunet "${predictions}" --freetumor "${freetumor}" \
    --stunet-probabilities "${predictions}" --output "${analysis}" --workers 2 \
    --closed-only --case-cache-dir "${output}/closed_case_metrics" \
    >"${output}/closed_metrics.log" 2>&1
  "${python_bin}" - "${epoch}" "${analysis}" "${checkpoint}" "${checkpoint_sha}" "${evaluation_branch}" \
    "${output}/TERMINAL.json" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
epoch=int(sys.argv[1]);analysis_path=Path(sys.argv[2]);checkpoint=Path(sys.argv[3]);checkpoint_sha=sys.argv[4];branch=sys.argv[5];output=Path(sys.argv[6])
x=json.load(open(analysis_path));keys=("stunet_standalone","gf_anchor_confirm_either_p70")
if (x.get("status")!="COMPLETE" or x.get("case_count")!=50 or not x.get("closed_only")
    or set(x.get("metrics",{}))!={"anchor",*keys} or x.get("hidden_read")
    or x.get("submission_created") or x.get("uploaded")):
    raise SystemExit("invalid complete-hybrid closed public analysis")
rows=[]
for key in keys:
    metric=x["metrics"][key];dsc=float(metric["Lesion_DSC"]);nsd=float(metric["Lesion_NSD"])
    rows.append({"epoch":epoch,"configuration":key,"Lesion_DSC":dsc,"Lesion_NSD":nsd,
                 "public_slot_gate":dsc>=0.6366639257651584 and nsd>=0.5347520263176905})
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","evaluation_epoch":epoch,"branch":branch,
 "results":rows,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis_path.resolve()),"sha256":hashlib.sha256(analysis_path.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

for epoch in "${epochs[@]}"; do evaluate_epoch "${epoch}"; done

PYTHONPATH="${root}" "${python_bin}" - "${output_root}/epoch_0040/TERMINAL.json" \
  "${output_root}/epoch_0080/TERMINAL.json" "${training_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
from select_complete_hybrid_public import select_candidate
e40,e80,training,output=map(Path,sys.argv[1:]);rows=[]
for path in (e40,e80):
    payload=json.load(open(path));
    if payload.get("status")!="COMPLETE" or payload.get("hidden_read"): raise SystemExit("invalid endpoint terminal")
    for row in payload["results"]:
        rows.append({**row,"terminal":{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()},
                     "checkpoint":payload["checkpoint"],"analysis":payload["analysis"]})
selected,qualified=select_candidate(rows)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REVIEW_TRAINED_CANDIDATE" if qualified else "NO_HIDDEN_ARCHIVE",
 "predeclared_epochs":[40,80],"predeclared_configurations":["stunet_standalone","gf_anchor_confirm_either_p70"],
 "results":rows,"selected":selected,
 "selection_rule":"maximum public DSC among configurations satisfying DSC>=0.6366639257651584 and NSD>=0.5347520263176905; NSD tie-break",
 "training_terminal":{"path":str(training.resolve()),"sha256":hashlib.sha256(training.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT
