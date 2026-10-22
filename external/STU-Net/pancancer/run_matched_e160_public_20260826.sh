#!/usr/bin/env bash
set -euo pipefail

# Material trained-model checkpoint comparison. This controller is deliberately
# public-only: it has no hidden-data path, packaging command, or upload client.
# Training is paused only after both immutable epoch-160 inference snapshots
# are stable, and is resumed from checkpoint_latest on every exit path.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

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
base_checkpoint="${base_fold}/checkpoint_inference_epoch_0160.pth"
base_sidecar="${base_fold}/checkpoint_inference_epoch_0160.json"
base_latest="${base_fold}/checkpoint_latest.pth"
component_checkpoint="${component_fold}/checkpoint_inference_epoch_0160.pth"
component_sidecar="${component_fold}/checkpoint_inference_epoch_0160.json"
component_latest="${component_fold}/checkpoint_latest.pth"
component_manifest="${root}/runs/stunetb_componentbox_e100_to300_20260826/SOURCE.json"
base_unit="miccai-stunetb-base-gpu0-post-e100-20260826.service"
component_unit="miccai-stunetb-component-gpu1-post-e100-20260826.service"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
output="${root}/public_sweeps/stunetb_matched_e160_20260826"
base_predictions="${output}/base_predictions"
component_predictions="${output}/component_predictions"
base_analysis="${output}/base_ensemble.json"
component_analysis="${output}/component_ensemble.json"
pause_marker="${output}/TRAINING_PAUSED.json"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
analysis_workers="${STUNET_ANALYSIS_WORKERS:-1}"

[[ "${analysis_workers}" =~ ^[1-2]$ ]] || {
  echo "STUNET_ANALYSIS_WORKERS must be 1-2 for the memory-bounded full-volume evaluation" >&2
  exit 2
}
for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${output}" "${base_predictions}" "${component_predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "matched epoch-160 public controller already active" >&2; exit 1; }

training_paused=false
failure_line=unknown
resume_branches() {
  /usr/bin/bash "${dual_launcher}"
}
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${output}" "${terminal}" <<'PY'
import datetime, json, os, sys
from pathlib import Path
reason=sys.argv[1]; root=Path(sys.argv[2]); output=Path(sys.argv[3])
public_images_read=any(root.glob("*_CASE_*.json"))
public_labels_read=(any(root.glob("*_ensemble_case_metrics/*.json"))
                    or (root/"base_ensemble.json").is_file()
                    or (root/"component_ensemble.json").is_file())
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"CONTINUE_TRAINING",
 "public_read":public_images_read or public_labels_read,
 "public_images_read":public_images_read,"public_labels_read":public_labels_read,
 "hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "matched epoch-160 public controller failed at line ${line}, status ${status}"
  fi
  if [[ "${training_paused}" == true ]]; then resume_branches || true; fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) resume_branches; trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected epoch-160 terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi

# Require both checkpoint files and their independently written metric
# sidecars to remain unchanged across three polls before stopping training.
stable=0
old_state=""
while :; do
  if [[ -s "${source}" && -s "${base_checkpoint}" && -s "${base_sidecar}" \
        && -s "${base_latest}" && -s "${component_checkpoint}" \
        && -s "${component_sidecar}" && -s "${component_latest}" \
        && -s "${component_manifest}" ]]; then
    state=$(stat -c '%s:%Y' "${base_checkpoint}" "${base_sidecar}" \
      "${component_checkpoint}" "${component_sidecar}" | tr '\n' ':')
    if [[ "${state}" == "${old_state}" ]]; then stable=$((stable+1)); else stable=0; old_state="${state}"; fi
    if [[ "${stable}" -ge 3 ]]; then
      "${python_bin}" - "${base_checkpoint}" "${base_sidecar}" \
        "${component_checkpoint}" "${component_sidecar}" <<'PY'
import json, os, sys
from pathlib import Path
for role, checkpoint, sidecar, trainer in (
    ("base", Path(sys.argv[1]), Path(sys.argv[2]), "STUNetTrainerPanCancer"),
    ("component", Path(sys.argv[3]), Path(sys.argv[4]), "STUNetTrainerPanCancerComponentBox"),
):
    payload=json.load(open(sidecar,encoding="utf-8"))
    if (payload.get("schema_version") != 1 or payload.get("current_epoch") != 160
        or payload.get("checkpoint_kind") != "inference_only"
        or payload.get("checkpoint") != checkpoint.name
        or payload.get("checkpoint_bytes") != checkpoint.stat().st_size
        or payload.get("trainer_name") != trainer):
        raise SystemExit(f"invalid {role} epoch-160 inference checkpoint sidecar")
PY
      break
    fi
  fi
  printf '%s waiting for stable matched epoch-160 inference checkpoints\n' "$(date -Ins)"
  sleep 20
done

base_sha=$(sha256sum "${base_checkpoint}" | awk '{print $1}')
component_sha=$(sha256sum "${component_checkpoint}" | awk '{print $1}')
for unit in "${base_unit}" "${component_unit}"; do
  if systemctl --user is-active --quiet "${unit}"; then systemctl --user stop "${unit}"; fi
done
training_paused=true
while systemctl --user is-active --quiet "${base_unit}" \
   || systemctl --user is-active --quiet "${component_unit}"; do sleep 2; done
[[ "${base_sha}" == "$(sha256sum "${base_checkpoint}" | awk '{print $1}')" ]]
[[ "${component_sha}" == "$(sha256sum "${component_checkpoint}" | awk '{print $1}')" ]]

base_resume_validation=$("${python_bin}" "${validator}" --mode base \
  --source "${source}" --latest "${base_latest}")
component_resume_validation=$("${python_bin}" "${validator}" --mode component \
  --source "${source}" --latest "${component_latest}" \
  --trainer-source "${trainer_source}" --manifest "${component_manifest}")
base_resume_sha=$(sha256sum "${base_latest}" | awk '{print $1}')
component_resume_sha=$(sha256sum "${component_latest}" | awk '{print $1}')

"${python_bin}" - "${base_checkpoint}" "${base_sha}" "${component_checkpoint}" \
  "${component_sha}" "${base_latest}" "${base_resume_sha}" \
  "${component_latest}" "${component_resume_sha}" "${base_resume_validation}" \
  "${component_resume_validation}" "${pause_marker}" <<'PY'
import datetime, json, os, sys
from pathlib import Path
args=sys.argv[1:]; output=Path(args[-1])
paths=[Path(args[i]) for i in (0,2,4,6)]; hashes=[args[i] for i in (1,3,5,7)]
base_resume=json.loads(args[8]); component_resume=json.loads(args[9])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "reason":"matched epoch-160 fixed-public comparison of two materially trained branches",
 "evaluation_epoch":160,
 "base_checkpoint":{"path":str(paths[0].resolve()),"sha256":hashes[0]},
 "component_checkpoint":{"path":str(paths[1].resolve()),"sha256":hashes[1]},
 "base_resume":{"path":str(paths[2].resolve()),"sha256":hashes[2],"epoch":base_resume["latest_epoch"]},
 "component_resume":{"path":str(paths[3].resolve()),"sha256":hashes[3],"epoch":component_resume["latest_epoch"]},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
     && "${used0}" -le 4096 && "${used1}" -le 4096 ]] && break
  sleep 10
done

predict_casewise() {
  local model="$1" checkpoint_name="$2" predictions="$3" prefix="$4" gpu="$5"
  local part_id part4
  for part_id in $(seq 0 49); do
    part4=$(printf '%04d' "${part_id}")
    "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
      --model-dir "${model}" --checkpoint "${checkpoint_name}" --gpu "${gpu}" \
      --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities --sequential \
      --dataset-role public --terminal "${output}/${prefix}_CASE_${part4}.json" \
      >"${output}/${prefix,,}_case_${part4}.log" 2>&1
  done
}

prediction_set_complete() {
  local predictions="$1" prefix="$2" checkpoint_sha="$3"
  "${python_bin}" - "${predictions}" "${output}" "${prefix}" "${checkpoint_sha}" <<'PY'
import json, sys
from pathlib import Path
predictions, root, prefix = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
checkpoint_sha = sys.argv[4]
masks = sorted(predictions.glob("*.nii.gz"))
probabilities = sorted(predictions.glob("*.prob16.npz"))
terminals = sorted(root.glob(f"{prefix}_CASE_*.json"))
if not (len(masks) == len(probabilities) == len(terminals) == 50):
    raise SystemExit(1)
mask_names = {path.name for path in masks if path.stat().st_size > 0}
probability_names = {path.name for path in probabilities if path.stat().st_size > 0}
cases = set()
for terminal in terminals:
    try:
        payload = json.load(open(terminal, encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        raise SystemExit(1)
    if (payload.get("status") != "COMPLETE"
        or payload.get("dataset_role") != "public"
        or payload.get("hidden_read")
        or payload.get("checkpoint", {}).get("sha256") != checkpoint_sha
        or payload.get("case_count") != 1
        or len(payload.get("cases", [])) != 1
        or len(payload.get("probability_files", [])) != 1):
        raise SystemExit(1)
    case = payload["cases"][0]
    probability = payload["probability_files"][0]
    if case not in mask_names or probability not in probability_names:
        raise SystemExit(1)
    cases.add(case)
if cases != mask_names or len(cases) != 50:
    raise SystemExit(1)
PY
}

# Run only one full-volume predictor at a time and keep preprocessing/export in
# that same process. A spawned legacy export worker duplicated the full logits
# and was still enough for systemd-oomd to kill a serial predictor on a later
# large CT. Sequential export removes that cross-process full-volume copy while
# preserving nnU-Net's official conversion and probability archive path.
prediction_set_complete "${base_predictions}" BASE "${base_sha}" || \
  predict_casewise "${base_model}" "$(basename "${base_checkpoint}")" \
    "${base_predictions}" BASE 0
prediction_set_complete "${component_predictions}" COMPONENT "${component_sha}" || \
  predict_casewise "${component_model}" "$(basename "${component_checkpoint}")" \
    "${component_predictions}" COMPONENT 1

for predictions in "${base_predictions}" "${component_predictions}"; do
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  [[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
done
[[ "${base_sha}" == "$(sha256sum "${base_checkpoint}" | awk '{print $1}')" ]]
[[ "${component_sha}" == "$(sha256sum "${component_checkpoint}" | awk '{print $1}')" ]]

"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${base_predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${base_predictions}" --output "${base_analysis}" \
  --workers "${analysis_workers}" --closed-only \
  --case-cache-dir "${output}/base_closed_case_metrics" \
  >"${output}/base_ensemble.log" 2>&1
"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${component_predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${component_predictions}" --output "${component_analysis}" \
  --workers "${analysis_workers}" --closed-only \
  --case-cache-dir "${output}/component_closed_case_metrics" \
  >"${output}/component_ensemble.log" 2>&1

"${python_bin}" - "${base_analysis}" "${component_analysis}" "${pause_marker}" \
  "${summary}" "${base_checkpoint}" "${base_sha}" "${component_checkpoint}" \
  "${component_sha}" "${terminal}" <<'PY'
import csv, datetime, hashlib, json, os, sys
from pathlib import Path
base_path,component_path,pause_path,summary_path,base_checkpoint=map(Path,sys.argv[1:6])
base_sha=sys.argv[6]; component_checkpoint=Path(sys.argv[7]); component_sha=sys.argv[8]; output=Path(sys.argv[9])
analyses={}
rows=[]
for branch,path in (("base_global_box",base_path),("component_box",component_path)):
    analysis=json.load(open(path,encoding="utf-8"))
    if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
        or not analysis.get("public_labels_read") or analysis.get("hidden_read")
        or analysis.get("submission_created") or analysis.get("uploaded")):
        raise SystemExit(f"invalid {branch} public analysis")
    analyses[branch]=analysis
    rows.extend({"branch":branch,"checkpoint_epoch":160,**row} for row in analysis["deployments"])
rows.sort(key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]),reverse=True)
closed=[]
for branch,analysis in analyses.items():
    for configuration in ("stunet_standalone","gf_anchor_confirm_either_p70"):
        metric=analysis["metrics"][configuration]
        closed.append({"branch":branch,"checkpoint_epoch":160,"configuration":configuration,
                       **metric,"DSC_gain":metric["Lesion_DSC"]-analysis["metrics"]["anchor"]["Lesion_DSC"],
                       "NSD_gain":metric["Lesion_NSD"]-analysis["metrics"]["anchor"]["Lesion_NSD"]})
closed.sort(key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]),reverse=True)
qualified=[row for row in closed if row["Lesion_DSC"]>=0.6154316379500749 and row["Lesion_NSD"]>=0.5104225107769019]
fields=["branch","checkpoint_epoch","configuration","Lesion_DSC","Lesion_NSD","DSC_gain","NSD_gain"]
with open(summary_path,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=fields,delimiter="\t"); writer.writeheader(); writer.writerows(rows)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REVIEW_MATERIAL_CANDIDATE" if qualified else "CONTINUE_TRAINING",
 "evaluation_epoch":160,"comparison":"matched global-box versus component-box trained branches",
 "selection_policy":"closed standalone and fixed p70 confirmation reported separately; broad grid diagnostic only",
 "closed_candidates":closed,"qualified_closed_candidates":qualified,
 "best_diagnostic_observed":rows[0],
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "base_checkpoint":{"path":str(base_checkpoint.resolve()),"sha256":base_sha},
 "component_checkpoint":{"path":str(component_checkpoint.resolve()),"sha256":component_sha},
 "base_analysis":{"path":str(base_path.resolve()),"sha256":hashlib.sha256(base_path.read_bytes()).hexdigest()},
 "component_analysis":{"path":str(component_path.resolve()),"sha256":hashlib.sha256(component_path.read_bytes()).hexdigest()},
 "training_pause":{"path":str(pause_path.resolve()),"sha256":hashlib.sha256(pause_path.read_bytes()).hexdigest()},
 "summary":{"path":str(summary_path.resolve()),"sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

resume_branches
training_paused=false
trap - EXIT
cat "${summary}"
