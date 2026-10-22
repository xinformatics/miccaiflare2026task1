#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Predeclared matched public comparison 30 optimizer epochs after the e100 fork.
# This controller has no hidden path, archive operation, network upload, or
# Codabench client. Both training branches resume after the public decision.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${root}/predict_stunet_safe.py"
analyzer="${root}/analyze_stunet_gf_ensemble.py"
validator="${root}/validate_post_e100_branch.py"
dual_launcher="${root}/launch_stunetb_dual_after_e100_20260826.sh"
trainer_source="${root}/STUNetTrainerPanCancer.py"
base_model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
component_model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
base_fold="${base_model}/fold_all"
component_fold="${component_model}/fold_all"
source="${base_fold}/checkpoint_epoch_0100.pth"
base_checkpoint="${base_fold}/checkpoint_epoch_0130.pth"
base_latest="${base_fold}/checkpoint_latest.pth"
component_checkpoint="${component_fold}/checkpoint_epoch_0130.pth"
component_latest="${component_fold}/checkpoint_latest.pth"
component_manifest="${root}/runs/stunetb_componentbox_e100_to300_20260826/SOURCE.json"
base_unit="miccai-stunetb-base-gpu0-post-e100-20260826.service"
component_unit="miccai-stunetb-component-gpu1-post-e100-20260826.service"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
output="${root}/public_sweeps/stunetb_matched_e130_20260826"
base_predictions="${output}/base_predictions"
component_predictions="${output}/component_predictions"
base_analysis="${output}/base_ensemble.json"
component_analysis="${output}/component_ensemble.json"
pause_marker="${output}/TRAINING_PAUSED.json"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
compact_equivalence="${root}/public_sweeps/direct_compact_equivalence_e100_20260826/EQUIVALENCE.json"
analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"
[[ "${analysis_workers}" =~ ^[1-9][0-9]*$ && "${analysis_workers}" -le 4 ]] || {
  echo "STUNET_ANALYSIS_WORKERS must be 1-4 for memory-safe full-volume scoring" >&2
  exit 2
}

mkdir -p "${output}" "${base_predictions}" "${component_predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "matched epoch-130 public controller already active" >&2; exit 1; }

resume_branches() {
  /usr/bin/bash "${dual_launcher}"
}

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"KEEP_TOUCHCOMP",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "matched epoch-130 public controller failed at line ${line}, status ${status}"
  fi
  if [[ "${status}" -ne 0 ]]; then resume_branches || true; fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) resume_branches;trap - EXIT;exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected matched epoch-130 terminal: ${existing}" >&2;exit 2 ;;
  esac
fi

stable=0
old_state=""
while :; do
  if [[ -s "${source}" && -s "${base_checkpoint}" && -s "${base_latest}" \
        && -s "${component_checkpoint}" && -s "${component_latest}" \
        && -s "${component_manifest}" ]]; then
    state=$(stat -c '%s:%Y' "${base_checkpoint}" "${base_latest}" \
      "${component_checkpoint}" "${component_latest}" | tr '\n' ':')
    if [[ "${state}" == "${old_state}" ]]; then stable=$((stable+1));else stable=0;old_state="${state}";fi
    if [[ "${stable}" -ge 3 ]]; then
      base_checkpoint_validation=$("${python_bin}" "${validator}" --mode base \
        --source "${source}" --latest "${base_checkpoint}")
      "${python_bin}" "${validator}" --mode base --source "${source}" --latest "${base_latest}" >/dev/null
      component_checkpoint_validation=$("${python_bin}" "${validator}" --mode component --source "${source}" \
        --latest "${component_checkpoint}" --trainer-source "${trainer_source}" \
        --manifest "${component_manifest}")
      "${python_bin}" "${validator}" --mode component --source "${source}" \
        --latest "${component_latest}" --trainer-source "${trainer_source}" \
        --manifest "${component_manifest}" >/dev/null
      base_epoch=$("${python_bin}" -c 'import json,sys;print(json.loads(sys.argv[1])["latest_epoch"])' "${base_checkpoint_validation}")
      component_epoch=$("${python_bin}" -c 'import json,sys;print(json.loads(sys.argv[1])["latest_epoch"])' "${component_checkpoint_validation}")
      [[ "${base_epoch}" -eq 130 && "${component_epoch}" -eq 130 ]] && break
    fi
  fi
  printf '%s waiting for stable matched epoch-130 checkpoints\n' "$(date -Ins)"
  sleep 20
done

base_sha=$(sha256sum "${base_checkpoint}" | awk '{print $1}')
component_sha=$(sha256sum "${component_checkpoint}" | awk '{print $1}')
base_latest_sha=$(sha256sum "${base_latest}" | awk '{print $1}')
component_latest_sha=$(sha256sum "${component_latest}" | awk '{print $1}')
for unit in "${base_unit}" "${component_unit}"; do
  if systemctl --user is-active --quiet "${unit}"; then systemctl --user stop "${unit}";fi
done
while systemctl --user is-active --quiet "${base_unit}" \
   || systemctl --user is-active --quiet "${component_unit}"; do sleep 2;done
[[ "${base_sha}" == "$(sha256sum "${base_checkpoint}" | awk '{print $1}')" ]]
[[ "${component_sha}" == "$(sha256sum "${component_checkpoint}" | awk '{print $1}')" ]]
[[ "${base_latest_sha}" == "$(sha256sum "${base_latest}" | awk '{print $1}')" ]]
[[ "${component_latest_sha}" == "$(sha256sum "${component_latest}" | awk '{print $1}')" ]]

"${python_bin}" - "${base_checkpoint}" "${base_sha}" "${base_latest}" "${base_latest_sha}" \
  "${component_checkpoint}" "${component_sha}" "${component_latest}" "${component_latest_sha}" \
  "${component_manifest}" "${pause_marker}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
args=sys.argv[1:];output=Path(args[-1]);manifest=Path(args[-2]);pairs=args[:-2]
assets=[]
for index in range(0,len(pairs),2):
    path=Path(pairs[index]);expected=pairs[index+1]
    observed=hashlib.sha256(path.read_bytes()).hexdigest()
    if observed!=expected: raise SystemExit(f"pause hash mismatch: {path}")
    assets.append({"path":str(path.resolve()),"sha256":observed})
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "reason":"predeclared matched STU-Net epoch-130 public comparison",
 "base_checkpoint":assets[0],"base_resume_checkpoint":assets[1],
 "component_checkpoint":assets[2],"component_resume_checkpoint":assets[3],
 "component_source_manifest":{"path":str(manifest.resolve()),"sha256":hashlib.sha256(manifest.read_bytes()).hexdigest()},
 "hidden_read":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
     && "${used0}" -le 4096 && "${used1}" -le 4096 ]] && break
  sleep 15
done
for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
done

compact_args=()
if [[ -s "${compact_equivalence}" ]] && "${python_bin}" - "${compact_equivalence}" "${source}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,source=map(Path,sys.argv[1:]);x=json.load(open(terminal,encoding="utf-8"));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (x.get("status")!="COMPLETE" or not x.get("activation_authorized")
    or not x.get("mask_equal") or not x.get("probability_uint16_equal") or not x.get("geometry_equal")
    or x.get("maximum_probability_code_difference")!=0
    or x.get("checkpoint",{}).get("sha256")!=sha(source)
    or x.get("public_labels_read") or x.get("hidden_read")
    or x.get("submission_created") or x.get("uploaded")):
    raise SystemExit(1)
PY
then compact_args=(--direct-compact-probabilities);fi

predict_casewise() {
  local model="$1" checkpoint_name="$2" predictions="$3" prefix="$4"
  local gpu part_id part4
  for part_id in $(seq 0 49); do
    gpu=$((part_id % 2))
    part4=$(printf '%04d' "${part_id}")
    "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
      --model-dir "${model}" --checkpoint "${checkpoint_name}" \
      --gpu "${gpu}" --part-id "${part_id}" --num-parts 50 --workers 1 \
      --save-probabilities "${compact_args[@]}" --dataset-role public \
      --terminal "${output}/${prefix}_CASE_${part4}.json" \
      >"${output}/${prefix,,}_case_${part4}.log" 2>&1
  done
}

# Bound full-volume export memory to one CT and release its process before the
# next case. Long-lived and two-case predictors were measured to trigger oomd.
predict_casewise "${base_model}" "$(basename "${base_checkpoint}")" \
  "${base_predictions}" BASE
predict_casewise "${component_model}" "$(basename "${component_checkpoint}")" \
  "${component_predictions}" COMPONENT
for predictions in "${base_predictions}" "${component_predictions}"; do
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  [[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
done

"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${base_predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${base_predictions}" --output "${base_analysis}" --workers "${analysis_workers}" \
  >"${output}/base_ensemble.log" 2>&1
"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${component_predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${component_predictions}" --output "${component_analysis}" --workers "${analysis_workers}" \
  >"${output}/component_ensemble.log" 2>&1

"${python_bin}" - "${base_analysis}" "${component_analysis}" "${pause_marker}" \
  "${summary}" "${base_checkpoint}" "${component_checkpoint}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
base_path,component_path,pause_path,summary_path,base_checkpoint,component_checkpoint,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
analyses={}
for branch,path in (("base_global_box",base_path),("component_box",component_path)):
    analysis=json.load(open(path,encoding="utf-8"))
    if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
        or analysis.get("hidden_read") or analysis.get("submission_created") or analysis.get("uploaded")):
        raise SystemExit(f"invalid {branch} public analysis")
    analyses[branch]=analysis
    rows.extend({"branch":branch,"checkpoint_epoch":130,**row} for row in analysis["deployments"])
rows.sort(key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]),reverse=True)
eligible=[row for row in rows if row["Lesion_DSC"]>=0.6154316379500749 and row["Lesion_NSD"]>=0.5104225107769019]
selected=eligible[0] if eligible else rows[0];gate=bool(eligible)
fields=["branch","checkpoint_epoch","configuration","Lesion_DSC","Lesion_NSD","DSC_gain","NSD_gain"]
with open(summary_path,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=fields,delimiter="\t");writer.writeheader();writer.writerows(rows)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION" if gate else "KEEP_TOUCHCOMP",
 "predeclared_epoch":130,"predeclared_epochs":[130],"comparison":"matched global-box versus component-box continuation",
 "selection_policy":"maximum fixed-public DSC subject to immutable DSC/NSD gates",
 "selected":selected if gate else None,"best_observed":selected,
 "selected_branch":selected["branch"] if gate else None,
 "selected_epoch":130 if gate else None,
 "selected_configuration":selected["configuration"] if gate else None,
 "selected_fixed_public_DSC":selected["Lesion_DSC"] if gate else None,
 "selected_fixed_public_NSD":selected["Lesion_NSD"] if gate else None,
 "checkpoint":{"path":str((component_checkpoint if selected["branch"]=="component_box" else base_checkpoint).resolve()),
               "sha256":sha(component_checkpoint if selected["branch"]=="component_box" else base_checkpoint)} if gate else None,
 "selected_analysis":{"path":str((component_path if selected["branch"]=="component_box" else base_path).resolve()),
                      "sha256":sha(component_path if selected["branch"]=="component_box" else base_path)} if gate else None,
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "base_analysis":{"path":str(base_path.resolve()),"sha256":sha(base_path)},
 "component_analysis":{"path":str(component_path.resolve()),"sha256":sha(component_path)},
 "training_pause":{"path":str(pause_path.resolve()),"sha256":sha(pause_path)},
 "summary":{"path":str(summary_path.resolve()),"sha256":sha(summary_path)},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT
cat "${summary}"
