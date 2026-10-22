#!/usr/bin/env bash
set -euo pipefail

# Evaluate the leading component-box branch at epoch 200 without interrupting
# that branch's continued training. GPU 0 hosts inference while its weaker
# global-box control is paused, then resumes before CPU metric scoring starts.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${root}/predict_stunet_safe.py"
analyzer="${root}/analyze_stunet_gf_ensemble.py"
validator="${root}/validate_post_e100_branch.py"
dual_launcher="${root}/launch_stunetb_dual_after_e100_20260826.sh"
component_model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
component_fold="${component_model}/fold_all"
checkpoint="${component_fold}/checkpoint_inference_epoch_0200.pth"
sidecar="${component_fold}/checkpoint_inference_epoch_0200.json"
source="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0100.pth"
base_latest="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_latest.pth"
base_unit="miccai-stunetb-base-gpu0-post-e100-20260826.service"
component_unit="miccai-stunetb-component-gpu1-post-e100-20260826.service"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
e160_analysis="${root}/public_sweeps/stunetb_matched_e160_20260826/component_ensemble.json"
output="${root}/public_sweeps/stunetb_component_e200_20260826"
predictions="${output}/predictions"
analysis="${output}/closed_metrics.json"
pause_marker="${output}/BASE_TRAINING_PAUSED.json"
terminal="${output}/TERMINAL.json"

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2
    exit 2
  }
done
[[ -s "${e160_analysis}" ]]
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "component epoch-200 public controller already active" >&2; exit 1; }

base_paused=false
failure_line=unknown
resume_base() { /usr/bin/bash "${dual_launcher}"; }
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"CONTINUE_TRAINING",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "component epoch-200 public controller failed at line ${line}, status ${status}"
  fi
  if [[ "${base_paused}" == true ]]; then resume_base || true; fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) resume_base; trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected epoch-200 terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi

# Require the compact inference snapshot and sidecar to remain unchanged over
# three polls. Component training remains active throughout this wait.
stable=0
old_state=""
while :; do
  if [[ -s "${checkpoint}" && -s "${sidecar}" && -s "${source}" && -s "${base_latest}" ]]; then
    state=$(stat -c '%s:%Y' "${checkpoint}" "${sidecar}" | tr '\n' ':')
    if [[ "${state}" == "${old_state}" ]]; then stable=$((stable+1)); else stable=0; old_state="${state}"; fi
    if [[ "${stable}" -ge 3 ]]; then
      "${python_bin}" - "${checkpoint}" "${sidecar}" <<'PY'
import json,sys
from pathlib import Path
checkpoint,sidecar=map(Path,sys.argv[1:])
p=json.load(open(sidecar,encoding="utf-8"))
if (p.get("schema_version")!=1 or p.get("current_epoch")!=200
    or p.get("checkpoint_kind")!="inference_only"
    or p.get("checkpoint")!=checkpoint.name
    or p.get("checkpoint_bytes")!=checkpoint.stat().st_size
    or p.get("trainer_name")!="STUNetTrainerPanCancerComponentBox"):
    raise SystemExit("invalid component epoch-200 inference sidecar")
PY
      break
    fi
  fi
  printf '%s waiting for stable component epoch-200 inference checkpoint\n' "$(date -Ins)"
  sleep 20
done

checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
if systemctl --user is-active --quiet "${base_unit}"; then
  systemctl --user stop "${base_unit}"
fi
base_paused=true
while systemctl --user is-active --quiet "${base_unit}"; do sleep 2; done
systemctl --user is-active --quiet "${component_unit}"
[[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]
base_resume_validation=$("${python_bin}" "${validator}" --mode base --source "${source}" --latest "${base_latest}")
base_resume_sha=$(sha256sum "${base_latest}" | awk '{print $1}')

"${python_bin}" - "${checkpoint}" "${checkpoint_sha}" "${base_latest}" \
  "${base_resume_sha}" "${base_resume_validation}" "${pause_marker}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
checkpoint=Path(sys.argv[1]); checkpoint_sha=sys.argv[2]
latest=Path(sys.argv[3]); latest_sha=sys.argv[4]
resume=json.loads(sys.argv[5]); output=Path(sys.argv[6])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "reason":"component-box epoch-200 public inference uses GPU 0 while component training continues",
 "evaluation_checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha,"epoch":200},
 "base_resume":{"path":str(latest.resolve()),"sha256":latest_sha,"epoch":resume["latest_epoch"]},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  sleep 10
done

for part_id in $(seq 0 49); do
  part4=$(printf '%04d' "${part_id}")
  "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
    --model-dir "${component_model}" --checkpoint "$(basename "${checkpoint}")" --gpu 0 \
    --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities --sequential \
    --dataset-role public --terminal "${output}/E200_COMPONENT_CASE_${part4}.json" \
    >"${output}/case_${part4}.log" 2>&1
done

[[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
[[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]

# Return GPU 0 to training before the CPU-only exact metrics pass.
resume_base
base_paused=false

"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${predictions}" --output "${analysis}" \
  --workers 2 --closed-only --case-cache-dir "${output}/closed_case_metrics" \
  >"${output}/closed_metrics.log" 2>&1

"${python_bin}" - "${analysis}" "${e160_analysis}" "${checkpoint}" \
  "${checkpoint_sha}" "${pause_marker}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
analysis_path,e160_path,checkpoint=map(Path,sys.argv[1:4])
checkpoint_sha=sys.argv[4]; pause_path=Path(sys.argv[5]); output=Path(sys.argv[6])
current=json.load(open(analysis_path,encoding="utf-8"))
previous=json.load(open(e160_path,encoding="utf-8"))
if (current.get("status")!="COMPLETE" or current.get("case_count")!=50
    or not current.get("closed_only") or current.get("hidden_read")
    or current.get("submission_created") or current.get("uploaded")):
    raise SystemExit("invalid epoch-200 closed public analysis")
keys=("stunet_standalone","gf_anchor_confirm_either_p70")
deltas={key:{metric:current["metrics"][key][metric]-previous["metrics"][key][metric]
             for metric in ("Lesion_DSC","Lesion_NSD")} for key in keys}
p70=current["metrics"]["gf_anchor_confirm_either_p70"]
standalone_gain=deltas["stunet_standalone"]["Lesion_DSC"]
p70_gain=deltas["gf_anchor_confirm_either_p70"]["Lesion_DSC"]
numerical_gate=p70["Lesion_DSC"]>=0.6154316379500749 and p70["Lesion_NSD"]>=0.5104225107769019
material_gain=standalone_gain>=0.04 or p70_gain>=0.01
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REVIEW_TRAINED_CANDIDATE" if numerical_gate and material_gain else "CONTINUE_TRAINING",
 "evaluation_epoch":200,"branch":"component_box","metrics":current["metrics"],
 "epoch200_minus_epoch160":deltas,"high_value_public_diagnostic":current["high_value_public_diagnostic"],
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "material_gain_rule":"standalone DSC +0.04 or fixed-p70 DSC +0.01 versus component epoch 160",
 "numerical_gate":numerical_gate,"material_trained_gain":material_gain,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis_path.resolve()),"sha256":hashlib.sha256(analysis_path.read_bytes()).hexdigest()},
 "epoch160_analysis":{"path":str(e160_path.resolve()),"sha256":hashlib.sha256(e160_path.read_bytes()).hexdigest()},
 "base_training_pause":{"path":str(pause_path.resolve()),"sha256":hashlib.sha256(pause_path.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

trap - EXIT
