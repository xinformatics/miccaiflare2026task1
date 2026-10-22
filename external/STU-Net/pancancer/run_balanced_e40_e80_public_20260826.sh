#!/usr/bin/env bash
set -euo pipefail

# Predeclared mid/end-point evaluation of the epoch-160 balanced continuation.
# GPU 1 is used only after the original component epoch-300 evaluator exits.
# The default final path never pauses GPU 0 training; explicit interim-e40 mode
# requires the trainer to be externally SIGSTOP-frozen. No hidden/archive path
# exists here.
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
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160Balanced__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold="${model}/fold_all"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
e160_analysis="${root}/public_sweeps/stunetb_matched_e160_20260826/component_ensemble.json"
e300_terminal="${root}/public_sweeps/stunetb_component_e300_20260826/TERMINAL.json"
e300_deferred="${root}/public_sweeps/stunetb_component_e300_20260826/E300_DEFERRED_RESOURCE_PRESSURE.json"
training_manifest="${root}/runs/stunetb_componentbox_e160_balanced_80ep_20260826/BASE_CONTROL_REPURPOSED.json"
training_terminal="${root}/runs/stunetb_componentbox_e160_balanced_80ep_20260826/TRAINING_TERMINAL.json"
output_root="${root}/public_sweeps/stunetb_balanced_e40_e80_20260826"
terminal="${output_root}/TERMINAL.json"
epochs=(40 80)
interim_e40_only="${BALANCED_INTERIM_E40_ONLY:-0}"
[[ "${interim_e40_only}" == 0 || "${interim_e40_only}" == 1 ]] || {
  echo "BALANCED_INTERIM_E40_ONLY must be 0 or 1" >&2
  exit 2
}
interim_terminal="${output_root}/E40_INTERIM_COMPLETE.json"
failure_terminal="${terminal}"
if [[ "${interim_e40_only}" == 1 ]]; then
  epochs=(40)
  failure_terminal="${output_root}/E40_INTERIM_FAILED.json"
fi

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2
    exit 2
  }
done
[[ -s "${e160_analysis}" ]]
[[ -s "${training_manifest}" ]]
mkdir -p "${output_root}"
exec 9>"${output_root}/controller.lock"
flock -n 9 || { echo "balanced epoch-40/80 public controller already active" >&2; exit 1; }

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${failure_terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"REVIEW_BALANCED_PUBLIC_CONTROLLER",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${failure_terminal}" ]]; then
    write_failure "balanced public controller failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ "${interim_e40_only}" == 0 && -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output_root}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected balanced public terminal state: ${existing}" >&2; exit 2 ;;
  esac
fi

# Full-volume inference remains serialized with the already running endpoint
# audit. Epoch-300 metrics are not a selector for this predeclared run.
while [[ ! -s "${e300_terminal}" && ! -s "${e300_deferred}" ]]; do
  printf '%s waiting for original component epoch-300 public terminal\n' "$(date -Ins)"
  sleep 20
done
"${python_bin}" - "${e300_terminal}" "${e300_deferred}" <<'PY'
import json,sys
from pathlib import Path
terminal,deferred=map(Path,sys.argv[1:])
if terminal.is_file():
    p=json.load(open(terminal,encoding="utf-8"))
    if p.get("status") != "COMPLETE" or p.get("evaluation_epoch") != 300:
        raise SystemExit("original component epoch-300 public evaluation is not complete")
else:
    p=json.load(open(deferred,encoding="utf-8"))
    if (p.get("status")!="DEFERRED" or p.get("evaluation_epoch")!=300
        or p.get("decision")!="PRIORITIZE_BALANCED_TRAINING_SERIAL_COMPUTE"
        or p.get("completed_case_count")!=24
        or p.get("checkpoint",{}).get("sha256")!="be4b2872f74c58b8d972043f85d6d1c23ad9de4a57cb988bafad251797222fbb"):
        raise SystemExit("invalid component epoch-300 resource deferral")
if p.get("hidden_read") or p.get("submission_created") or p.get("uploaded"):
    raise SystemExit("invalid original component epoch-300 provenance")
PY

# Host swap is exhausted and concurrent training plus full-volume inference
# triggered two systemd-oomd kills. The default path therefore waits for a
# clean training exit. The explicit interim mode is used only while the exact
# trainer process group is externally SIGSTOP-frozen; it evaluates immutable
# epoch 40 and leaves the final two-endpoint terminal untouched.
if [[ "${interim_e40_only}" == 0 ]]; then
  while [[ ! -s "${training_terminal}" ]]; do
    printf '%s waiting for balanced training terminal before public inference\n' "$(date -Ins)"
    sleep 20
  done
  "${python_bin}" - "${training_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("status")!="COMPLETE" or p.get("public_read") or p.get("hidden_read"):
    raise SystemExit("balanced training did not complete with label-role-safe provenance")
if p.get("submission_created") or p.get("uploaded"):
    raise SystemExit("invalid balanced training provenance")
PY
fi

evaluate_epoch() {
  local epoch="$1" epoch4 output predictions analysis checkpoint sidecar checkpoint_sha stable old_state part_id part4 used case_terminal
  epoch4=$(printf '%04d' "${epoch}")
  output="${output_root}/epoch_${epoch4}"
  predictions="${output}/predictions"
  analysis="${output}/closed_metrics.json"
  checkpoint="${fold}/checkpoint_inference_epoch_${epoch4}.pth"
  sidecar="${fold}/checkpoint_inference_epoch_${epoch4}.json"
  mkdir -p "${output}" "${predictions}"

  stable=0
  old_state=""
  while :; do
    if [[ -s "${checkpoint}" && -s "${sidecar}" ]]; then
      state=$(stat -c '%s:%Y' "${checkpoint}" "${sidecar}" | tr '\n' ':')
      if [[ "${state}" == "${old_state}" ]]; then stable=$((stable+1)); else stable=0; old_state="${state}"; fi
      if [[ "${stable}" -ge 3 ]]; then
        "${python_bin}" - "${epoch}" "${checkpoint}" "${sidecar}" <<'PY'
import json,sys
from pathlib import Path
epoch=int(sys.argv[1]); checkpoint=Path(sys.argv[2]); sidecar=Path(sys.argv[3])
p=json.load(open(sidecar,encoding="utf-8"))
if (p.get("schema_version")!=1 or p.get("current_epoch")!=epoch
    or p.get("checkpoint_kind")!="inference_only"
    or p.get("checkpoint")!=checkpoint.name
    or p.get("checkpoint_bytes")!=checkpoint.stat().st_size
    or p.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160Balanced"):
    raise SystemExit(f"invalid balanced epoch-{epoch} inference sidecar")
PY
        break
      fi
    fi
    printf '%s waiting for stable balanced epoch-%s checkpoint\n' "$(date -Ins)" "${epoch}"
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
    case_terminal="${output}/BALANCED_E${epoch4}_CASE_${part4}.json"
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
      printf '%s reusing validated balanced epoch-%s public shard %s\n' "$(date -Ins)" "${epoch}" "${part_id}"
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

  "${python_bin}" - "${epoch}" "${analysis}" "${e160_analysis}" \
    "${checkpoint}" "${checkpoint_sha}" "${output}/TERMINAL.json" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
epoch=int(sys.argv[1]); analysis_path=Path(sys.argv[2]); reference_path=Path(sys.argv[3])
checkpoint=Path(sys.argv[4]); checkpoint_sha=sys.argv[5]; output=Path(sys.argv[6])
current=json.load(open(analysis_path,encoding="utf-8"))
reference=json.load(open(reference_path,encoding="utf-8"))
if (current.get("status")!="COMPLETE" or current.get("case_count")!=50
    or not current.get("closed_only") or current.get("hidden_read")
    or current.get("submission_created") or current.get("uploaded")):
    raise SystemExit(f"invalid balanced epoch-{epoch} closed public analysis")
keys=("stunet_standalone","gf_anchor_confirm_either_p70")
deltas={key:{metric:current["metrics"][key][metric]-reference["metrics"][key][metric]
             for metric in ("Lesion_DSC","Lesion_NSD")} for key in keys}
p70=current["metrics"]["gf_anchor_confirm_either_p70"]
gate=p70["Lesion_DSC"]>=0.6366639257651584 and p70["Lesion_NSD"]>=0.5347520263176905
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"PUBLIC_SLOT_GATE_MET" if gate else "KEEP_TRAINING_NO_HIDDEN_ARCHIVE",
 "evaluation_epoch":epoch,"branch":"component_box_e160_balanced","metrics":current["metrics"],
 "balanced_minus_original_component_e160":deltas,
 "high_value_public_diagnostic":current["high_value_public_diagnostic"],
 "fixed_public_DSC_slot_gate":0.6366639257651584,
 "fixed_public_NSD_floor":0.5347520263176905,"public_slot_gate":gate,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis_path.resolve()),"sha256":hashlib.sha256(analysis_path.read_bytes()).hexdigest()},
 "reference":{"path":str(reference_path.resolve()),"sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
}

for epoch in "${epochs[@]}"; do
  evaluate_epoch "${epoch}"
done

if [[ "${interim_e40_only}" == 1 ]]; then
  "${python_bin}" - "${output_root}/epoch_0040/TERMINAL.json" "${interim_terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload=json.load(open(source,encoding="utf-8"))
if (payload.get("status")!="COMPLETE" or payload.get("evaluation_epoch")!=40
    or payload.get("branch")!="component_box_e160_balanced"
    or payload.get("hidden_read") or payload.get("submission_created") or payload.get("uploaded")):
    raise SystemExit("invalid interim epoch-40 endpoint terminal")
result={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME",
 "epoch_terminal":{"path":str(source.resolve()),"sha256":hashlib.sha256(source.read_bytes()).hexdigest()},
 "metrics":payload["metrics"],"public_slot_gate":payload["public_slot_gate"],
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
print(json.dumps(result,indent=2))
PY
  trap - EXIT
  exit 0
fi

"${python_bin}" - "${output_root}/epoch_0040/TERMINAL.json" \
  "${output_root}/epoch_0080/TERMINAL.json" "${training_manifest}" \
  "${training_terminal}" "${e300_terminal}" "${e300_deferred}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
e40_path,e80_path,training_path,training_terminal,e300_path,deferred_path,output=map(Path,sys.argv[1:])
rows=[]
for path in (e40_path,e80_path):
    payload=json.load(open(path,encoding="utf-8"))
    if payload.get("status")!="COMPLETE" or payload.get("hidden_read"):
        raise SystemExit(f"invalid balanced endpoint terminal: {path}")
    metric=payload["metrics"]["gf_anchor_confirm_either_p70"]
    rows.append({"epoch":payload["evaluation_epoch"],**metric,
                 "public_slot_gate":payload["public_slot_gate"],
                 "terminal":{"path":str(path.resolve()),
                             "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}})
qualified=[row for row in rows if row["public_slot_gate"]]
# A higher-DSC endpoint that violates the NSD floor must never displace a
# slightly lower-DSC endpoint that satisfied both promotion requirements.
# Rank within the qualified set whenever it is nonempty; retain the best
# unqualified row only as diagnostic evidence when neither endpoint qualifies.
selected=max(qualified or rows,key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REVIEW_TRAINED_CANDIDATE" if qualified else "NO_HIDDEN_ARCHIVE",
 "predeclared_epochs":[40,80],"results":rows,"selected":selected,
 "selection_rule":"maximum fixed-p70 public DSC among endpoints satisfying DSC>=0.6366639257651584 and NSD>=0.5347520263176905, NSD tie-break; diagnostic maximum only if none qualify",
 "training_manifest":{"path":str(training_path.resolve()),
                      "sha256":hashlib.sha256(training_path.read_bytes()).hexdigest()},
 "training_terminal":{"path":str(training_terminal.resolve()),
                      "sha256":hashlib.sha256(training_terminal.read_bytes()).hexdigest()},
 "trajectory_dependency":{"path":str((e300_path if e300_path.is_file() else deferred_path).resolve()),
                          "sha256":hashlib.sha256((e300_path if e300_path.is_file() else deferred_path).read_bytes()).hexdigest()},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

trap - EXIT
