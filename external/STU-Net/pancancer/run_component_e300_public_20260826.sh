#!/usr/bin/env bash
set -euo pipefail

# Predeclared final-endpoint evaluation for the component-box STU-Net-B run.
# It waits for the epoch-200 decision to finish, then uses the now-free GPU 1
# while the global-box control can continue training on GPU 0.
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
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
checkpoint="${fold}/checkpoint_inference_epoch_0300.pth"
sidecar="${fold}/checkpoint_inference_epoch_0300.json"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
e160_analysis="${root}/public_sweeps/stunetb_matched_e160_20260826/component_ensemble.json"
e200_terminal="${root}/public_sweeps/stunetb_component_e200_20260826/TERMINAL.json"
output="${root}/public_sweeps/stunetb_component_e300_20260826"
predictions="${output}/predictions"
analysis="${output}/closed_metrics.json"
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
flock -n 9 || { echo "component epoch-300 public controller already active" >&2; exit 1; }

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"REVIEW_E300_CONTROLLER_FAILURE",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "component epoch-300 public controller failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected epoch-300 terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi

# Serialize the two full-volume evaluators. This is also a provenance gate:
# epoch 300 is interpreted only after the immutable epoch-200 result exists.
while [[ ! -s "${e200_terminal}" ]]; do
  printf '%s waiting for component epoch-200 public terminal\n' "$(date -Ins)"
  sleep 20
done
"${python_bin}" - "${e200_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("status") != "COMPLETE" or p.get("evaluation_epoch") != 200:
    raise SystemExit("epoch-200 public evaluation did not complete successfully")
if p.get("hidden_read") or p.get("submission_created") or p.get("uploaded"):
    raise SystemExit("invalid epoch-200 provenance")
PY

# Require the endpoint checkpoint and sidecar to remain unchanged for a full
# minute before hashing and inference.
stable=0
old_state=""
while :; do
  if [[ -s "${checkpoint}" && -s "${sidecar}" ]]; then
    state=$(stat -c '%s:%Y' "${checkpoint}" "${sidecar}" | tr '\n' ':')
    if [[ "${state}" == "${old_state}" ]]; then stable=$((stable+1)); else stable=0; old_state="${state}"; fi
    if [[ "${stable}" -ge 3 ]]; then
      "${python_bin}" - "${checkpoint}" "${sidecar}" <<'PY'
import json,sys
from pathlib import Path
checkpoint,sidecar=map(Path,sys.argv[1:])
p=json.load(open(sidecar,encoding="utf-8"))
if (p.get("schema_version")!=1 or p.get("current_epoch")!=300
    or p.get("checkpoint_kind")!="inference_only"
    or p.get("checkpoint")!=checkpoint.name
    or p.get("checkpoint_bytes")!=checkpoint.stat().st_size
    or p.get("trainer_name")!="STUNetTrainerPanCancerComponentBox"):
    raise SystemExit("invalid component epoch-300 inference sidecar")
PY
      break
    fi
  fi
  printf '%s waiting for stable component epoch-300 inference checkpoint\n' "$(date -Ins)"
  sleep 20
done

checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
while :; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  sleep 10
done

for part_id in $(seq 0 49); do
  part4=$(printf '%04d' "${part_id}")
  case_terminal="${output}/E300_COMPONENT_CASE_${part4}.json"
  if [[ -s "${case_terminal}" ]] && "${python_bin}" - \
    "${case_terminal}" "${predictions}" "${checkpoint_sha}" "${part_id}" <<'PY'
import json,sys
from pathlib import Path
terminal=Path(sys.argv[1]); predictions=Path(sys.argv[2])
expected_sha=sys.argv[3]; expected_part=int(sys.argv[4])
p=json.load(open(terminal,encoding="utf-8")); cases=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("dataset_role")!="public"
    or p.get("part_id")!=expected_part or p.get("num_parts")!=50
    or p.get("case_count")!=1 or len(cases)!=1
    or p.get("checkpoint",{}).get("sha256")!=expected_sha
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
name=cases[0]
mask=predictions/name
probability=predictions/f"{name.removesuffix('.nii.gz')}.prob16.npz"
if not mask.is_file() or mask.stat().st_size<1 or not probability.is_file() or probability.stat().st_size<1:
    raise SystemExit(1)
PY
  then
    printf '%s reusing validated component epoch-300 public shard %s\n' "$(date -Ins)" "${part_id}"
    continue
  fi
  "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
    --model-dir "${model}" --checkpoint "$(basename "${checkpoint}")" --gpu 1 \
    --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities --sequential \
    --dataset-role public --terminal "${case_terminal}" \
    >"${output}/case_${part4}.log" 2>&1
done

[[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
[[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]

"${python_bin}" -u "${analyzer}" --labels "${labels}" --anchor "${anchor}" \
  --stunet "${predictions}" --freetumor "${freetumor}" \
  --stunet-probabilities "${predictions}" --output "${analysis}" \
  --workers 2 --closed-only --case-cache-dir "${output}/closed_case_metrics" \
  >"${output}/closed_metrics.log" 2>&1

"${python_bin}" - "${analysis}" "${e160_analysis}" "${e200_terminal}" \
  "${checkpoint}" "${checkpoint_sha}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
analysis_path,e160_path,e200_path,checkpoint=map(Path,sys.argv[1:5])
checkpoint_sha=sys.argv[5]; output=Path(sys.argv[6])
current=json.load(open(analysis_path,encoding="utf-8"))
e160=json.load(open(e160_path,encoding="utf-8"))
e200=json.load(open(e200_path,encoding="utf-8"))
if (current.get("status")!="COMPLETE" or current.get("case_count")!=50
    or not current.get("closed_only") or current.get("hidden_read")
    or current.get("submission_created") or current.get("uploaded")):
    raise SystemExit("invalid epoch-300 closed public analysis")
keys=("stunet_standalone","gf_anchor_confirm_either_p70")
def deltas(reference):
    return {key:{metric:current["metrics"][key][metric]-reference["metrics"][key][metric]
                 for metric in ("Lesion_DSC","Lesion_NSD")} for key in keys}
d160=deltas(e160)
d200=deltas(e200)
p70=current["metrics"]["gf_anchor_confirm_either_p70"]
best_previous={key:{metric:max(e160["metrics"][key][metric],e200["metrics"][key][metric])
                    for metric in ("Lesion_DSC","Lesion_NSD")} for key in keys}
gain_over_best={key:{metric:current["metrics"][key][metric]-best_previous[key][metric]
                     for metric in ("Lesion_DSC","Lesion_NSD")} for key in keys}
numerical_gate=p70["Lesion_DSC"]>=0.6154316379500749 and p70["Lesion_NSD"]>=0.5104225107769019
material_gain=(gain_over_best["stunet_standalone"]["Lesion_DSC"]>=0.04
               or gain_over_best["gf_anchor_confirm_either_p70"]["Lesion_DSC"]>=0.01)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REVIEW_TRAINED_CANDIDATE" if numerical_gate and material_gain else "ENDPOINT_MEASURED_NO_HIDDEN_ARCHIVE",
 "evaluation_epoch":300,"branch":"component_box","metrics":current["metrics"],
 "epoch300_minus_epoch160":d160,"epoch300_minus_epoch200":d200,
 "epoch300_minus_best_previous":gain_over_best,
 "high_value_public_diagnostic":current["high_value_public_diagnostic"],
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "material_gain_rule":"standalone DSC +0.04 or fixed-p70 DSC +0.01 over the better of component epochs 160 and 200",
 "numerical_gate":numerical_gate,"material_trained_gain":material_gain,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis_path.resolve()),"sha256":hashlib.sha256(analysis_path.read_bytes()).hexdigest()},
 "epoch160_analysis":{"path":str(e160_path.resolve()),"sha256":hashlib.sha256(e160_path.read_bytes()).hexdigest()},
 "epoch200_terminal":{"path":str(e200_path.resolve()),"sha256":hashlib.sha256(e200_path.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

trap - EXIT
