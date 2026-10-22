#!/usr/bin/env bash
set -euo pipefail

# Public-only, sequential endpoint replay for the dual-window ResEnc S6.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${project_root}/nnunet/predict_resenc_s6_dualwindow_safe.py"
evaluator="${project_root}/nnunet/evaluate_resenc_s6_dualwindow_public.py"
selector="${project_root}/nnunet/select_resenc_s6_dualwindow_public.py"
policy="${project_root}/nnunet/RESENC_S6_DUALWINDOW_PUBLIC_SELECTION_POLICY_20260827.json"
selection_preflight="${project_root}/nnunet/RESENC_S6_DUALWINDOW_PREFLIGHT_20260827.json"
training_terminal="${project_root}/nnunet/runs/resenc_s6_dualwindow_f020_100ep_20260827/TRAINING_TERMINAL.json"
model="${project_root}/nnUNet_results/Dataset502_PanCancerPositiveUnlabeled/nnUNetTrainerResEncS6PanCancerDualWindow__nnUNetPlans__3d_fullres_ResEncS6"
fold="${model}/fold_all"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
output_root="${project_root}/nnunet/public_sweeps/resenc_s6_dualwindow_endpoints_20260827"
terminal="${output_root}/TERMINAL.json"
epochs=(25 50 100)

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${output_root}/.matplotlib"

mkdir -p "${output_root}"
exec 9>"${output_root}/controller.lock"
flock -n 9 || { echo "dual-window ResEnc public controller already active" >&2; exit 1; }

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RETRY_DUALWINDOW_RESENC_PUBLIC_ENDPOINTS",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  [[ "${status}" -eq 0 || -s "${terminal}" ]] || write_failure \
    "dual-window ResEnc public controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output_root}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected dual-window ResEnc public terminal status: ${status}" >&2; exit 2 ;;
  esac
fi

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "dual-window ResEnc fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
for file in "${predictor}" "${evaluator}" "${selector}" "${policy}" \
    "${selection_preflight}"; do
  [[ -s "${file}" ]] || { echo "missing dual-window ResEnc evaluator input: ${file}" >&2; exit 2; }
done
"${python_bin}" - "${project_root}" "${selection_preflight}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,record=map(Path,sys.argv[1:]);payload=json.load(open(record,encoding="utf-8"))
if (payload.get("status")!="PASS" or payload.get("tests",{}).get("total_passed")!=14
    or payload.get("tests",{}).get("total_failed")!=0
    or payload.get("public_read") or payload.get("hidden_read")):
    raise SystemExit("invalid ResEnc selection preflight")
for item in payload.get("code",{}).values():
    path=root/item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
        raise SystemExit(f"ResEnc selection-preflight hash mismatch: {path}")
PY

analysis_args=()
checkpoint_args=()
for epoch in "${epochs[@]}"; do
  epoch4=$(printf '%04d' "${epoch}")
  checkpoint_name="checkpoint_epoch_${epoch4}.pth"
  [[ "${epoch}" -eq 100 ]] && checkpoint_name="checkpoint_final.pth"
  checkpoint="${fold}/${checkpoint_name}"
  output="${output_root}/epoch_${epoch4}"
  predictions="${output}/predictions"
  analysis="${output}/metrics.json"
  endpoint_terminal="${output}/ENDPOINT_TERMINAL.json"
  mkdir -p "${predictions}"
  while [[ ! -s "${checkpoint}" ]]; do
    if [[ -s "${training_terminal}" ]]; then
      training_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${training_terminal}")
      [[ "${training_status}" != FAILED ]] || {
        echo "dual-window ResEnc training failed before endpoint ${epoch}" >&2
        exit 2
      }
    fi
    printf '%s waiting for dual-window ResEnc endpoint checkpoint %s\n' \
      "$(date -Ins)" "${epoch}"
    sleep 30
  done
  "${python_bin}" - "${checkpoint}" "${epoch}" <<'PY'
import sys,torch
from pathlib import Path
path=Path(sys.argv[1]);epoch=int(sys.argv[2]);x=torch.load(path,map_location="cpu",weights_only=False)
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerDualWindow"
    or int(x.get("current_epoch",-1))!=epoch
    or not isinstance(x.get("network_weights"),dict)):
    raise SystemExit(f"invalid dual-window ResEnc epoch-{epoch} checkpoint")
PY
  checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
  endpoint_valid=0
  if [[ -s "${endpoint_terminal}" ]] && "${python_bin}" - \
    "${endpoint_terminal}" "${analysis}" "${checkpoint_sha}" "${epoch}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,analysis=map(Path,sys.argv[1:3]);sha=sys.argv[3];epoch=int(sys.argv[4])
p=json.load(open(terminal,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("epoch")!=epoch
    or p.get("checkpoint",{}).get("sha256")!=sha
    or p.get("case_count")!=50 or p.get("dataset_role")!="public"
    or not p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded") or not analysis.is_file()
    or hashlib.sha256(analysis.read_bytes()).hexdigest()!=p.get("analysis",{}).get("sha256")):
    raise SystemExit(1)
PY
  then endpoint_valid=1; fi
  if [[ "${endpoint_valid}" -eq 1 ]]; then
    analysis_args+=(--analysis "${epoch}=${analysis}")
    checkpoint_args+=(--checkpoint "${epoch}=${checkpoint}")
    continue
  fi
  for part_id in $(seq 0 49); do
    part4=$(printf '%04d' "${part_id}")
    case_terminal="${output}/RESENC_E${epoch4}_CASE_${part4}.json"
    valid=0
    if [[ -s "${case_terminal}" ]] && "${python_bin}" - \
      "${case_terminal}" "${predictions}" "${checkpoint_sha}" "${part_id}" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"));pred=Path(sys.argv[2]);sha=sys.argv[3];part=int(sys.argv[4]);cases=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("trainer")!="nnUNetTrainerResEncS6PanCancerDualWindow"
    or p.get("dataset_role")!="public" or p.get("part_id")!=part or p.get("num_parts")!=50
    or p.get("case_count")!=1 or len(cases)!=1 or p.get("checkpoint",{}).get("sha256")!=sha
    or not p.get("direct_compact_probabilities") or p.get("public_labels_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
name=cases[0]
if not (pred/name).is_file() or not (pred/f"{name.removesuffix('.nii.gz')}.prob16.npz").is_file(): raise SystemExit(1)
PY
    then valid=1; fi
    [[ "${valid}" -eq 1 ]] && continue
    while :; do
      used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
      util=$(nvidia-smi --id=0 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
      [[ "${used}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ \
        && "${used}" -le 3072 && "${util}" -le 5 ]] && break
      sleep 15
    done
    "${python_bin}" -u "${predictor}" --images "${images}" --output "${predictions}" \
      --model-dir "${model}" --checkpoint "${checkpoint_name}" --gpu 0 \
      --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities \
      --direct-compact-probabilities --sequential --dataset-role public \
      --terminal "${case_terminal}" >"${output}/case_${part4}.log" 2>&1
  done
  [[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  [[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
  [[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]
  "${python_bin}" -u "${evaluator}" --policy "${policy}" --labels "${labels}" \
    --anchor "${anchor}" --resenc "${predictions}" --freetumor "${freetumor}" \
    --case-cache "${output}/case_metrics" --workers 2 --output "${analysis}" \
    >"${output}/metrics.log" 2>&1
  "${python_bin}" - "${checkpoint}" "${checkpoint_sha}" "${analysis}" \
    "${epoch}" "${endpoint_terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
checkpoint=Path(sys.argv[1]);checkpoint_sha=sys.argv[2];analysis=Path(sys.argv[3])
epoch=int(sys.argv[4]);output=Path(sys.argv[5]);payload=json.load(open(analysis,encoding="utf-8"))
if (payload.get("status")!="COMPLETE" or payload.get("case_count")!=50
    or payload.get("public_labels_read") is not True or payload.get("hidden_read")
    or payload.get("submission_created") or payload.get("uploaded")
    or len(payload.get("metrics",{}))!=4):
    raise SystemExit("invalid dual-window ResEnc endpoint analysis")
record={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","epoch":epoch,"dataset_role":"public","case_count":50,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis.resolve()),"sha256":hashlib.sha256(analysis.read_bytes()).hexdigest()},
 "public_labels_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
  analysis_args+=(--analysis "${epoch}=${analysis}")
  checkpoint_args+=(--checkpoint "${epoch}=${checkpoint}")
done

"${python_bin}" -u "${selector}" --policy "${policy}" \
  "${analysis_args[@]}" "${checkpoint_args[@]}" --output "${terminal}"
trap - EXIT
