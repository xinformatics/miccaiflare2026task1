#!/usr/bin/env bash
set -euo pipefail

# Public-only staged evaluator for the prospective partial-band branch.
# It has no hidden-image, archive, upload, or Codabench interface.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${project_root}/nnunet/predict_resenc_s6_partialband12_safe.py"
evaluator="${project_root}/nnunet/evaluate_resenc_s6_public.py"
selector="${project_root}/nnunet/select_resenc_s6_partialband12_public.py"
policy="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_PUBLIC_SELECTION_POLICY_20260827.json"
authorization="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_AUTHORIZATION_20260827.json"
preflight="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_TRAINER_PREFLIGHT_20260827.json"
controller_preflight="${project_root}/nnunet/RESENC_S6_PARTIALBAND12_CONTROLLER_PREFLIGHT_20260827.json"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/nnUNetTrainerResEncS6PanCancerPartialBand12__nnUNetPlans__3d_fullres_ResEncS6"
fold="${model}/fold_all"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
output_root="${project_root}/nnunet/public_sweeps/resenc_s6_partialband12_endpoints_20260827"
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
flock -n 9 || { echo "partial-band public controller already active" >&2; exit 1; }

failure_line=unknown
write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1];output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,"decision":"RETRY_PARTIALBAND12_PUBLIC",
 "public_labels_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  [[ "${status}" -eq 0 || -s "${terminal}" ]] || write_failure \
    "partial-band public controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output_root}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected partial-band public terminal status: ${status}" >&2;exit 2 ;;
  esac
fi

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2;exit 2;
  }
done
for file in "${predictor}" "${evaluator}" "${selector}" "${policy}" \
  "${authorization}" "${preflight}" "${controller_preflight}"; do
  [[ -s "${file}" ]] || { echo "missing partial-band public input: ${file}" >&2;exit 2; }
done

"${python_bin}" - "${project_root}" "${controller_preflight}" "${authorization}" "${preflight}" "${policy}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);controller_path,auth_path,preflight_path,policy_path=map(Path,sys.argv[2:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
controller=json.load(open(controller_path,encoding="utf-8"));safety=controller.get("safety",{})
if (controller.get("status")!="PASS_ARMED"
    or safety.get("current_authorization_satisfies_launch") is not True
    or safety.get("current_trainer_preflight_satisfies_launch") is not True):
    raise SystemExit("partial-band controller preflight is not armed")
for name,item in {
    "training controller":controller.get("controllers",{}).get("training",{}),
    "public controller":controller.get("controllers",{}).get("public",{}),
    "parent selector":controller.get("parent_selector",{}),
    "trainer preflight":controller.get("trainer_preflight",{}),
    "authorization":controller.get("authorization",{}),
}.items():
    path=(root/item.get("path","")).resolve()
    if not path.is_file() or sha(path)!=item.get("sha256"):
        raise SystemExit(f"partial-band runtime binding changed: {name}")
auth,preflight,policy=(json.load(open(path,encoding="utf-8")) for path in (auth_path,preflight_path,policy_path))
if (auth.get("status")!="AUTHORIZED_AFTER_BASE_SELECTION"
    or auth.get("decision")!="RUN_ONE_CONTROLLED_PARTIALBAND12_CONTINUATION"):
    raise SystemExit("partial-band public authorization is not armed")
if (preflight.get("status")!="PASS_ARMED"
    or preflight.get("authorization",{}).get("public_controller_implemented") is not True
    or preflight.get("authorization",{}).get("queue_armed") is not True):
    raise SystemExit("partial-band public preflight is not armed")
if (policy.get("checkpoint_epochs")!=[25,50,100]
    or policy.get("threshold_search_after_public_read") is not False
    or policy.get("hidden_read")):
    raise SystemExit("partial-band public policy changed")
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
  while [[ ! -s "${checkpoint}" ]]; do sleep 30; done
  "${python_bin}" - "${checkpoint}" "${epoch}" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location="cpu",weights_only=False);epoch=int(sys.argv[2])
if (x.get("trainer_name")!="nnUNetTrainerResEncS6PanCancerPartialBand12"
    or int(x.get("current_epoch",-1))!=epoch
    or not isinstance(x.get("network_weights"),dict)):
    raise SystemExit("invalid partial-band endpoint checkpoint")
PY
  checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
  endpoint_valid=0
  if [[ -s "${endpoint_terminal}" ]] && "${python_bin}" - \
    "${endpoint_terminal}" "${analysis}" "${checkpoint_sha}" "${epoch}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,analysis=map(Path,sys.argv[1:3]);sha=sys.argv[3];epoch=int(sys.argv[4]);p=json.load(open(terminal))
if (p.get("status")!="COMPLETE" or p.get("epoch")!=epoch or p.get("case_count")!=50
    or p.get("checkpoint",{}).get("sha256")!=sha or not p.get("public_labels_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")
    or not analysis.is_file() or p.get("analysis",{}).get("sha256")!=hashlib.sha256(analysis.read_bytes()).hexdigest()):
    raise SystemExit(1)
PY
  then endpoint_valid=1;fi
  if [[ "${endpoint_valid}" -eq 0 ]]; then
    for part_id in $(seq 0 49); do
      part4=$(printf '%04d' "${part_id}")
      case_terminal="${output}/BAND12_E${epoch4}_CASE_${part4}.json"
      valid=0
      if [[ -s "${case_terminal}" ]] && "${python_bin}" - \
        "${case_terminal}" "${predictions}" "${checkpoint_sha}" "${part_id}" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1]));pred=Path(sys.argv[2]);sha=sys.argv[3];part=int(sys.argv[4]);cases=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("trainer")!="nnUNetTrainerResEncS6PanCancerPartialBand12"
    or p.get("dataset_role")!="public" or p.get("part_id")!=part or p.get("num_parts")!=50
    or p.get("case_count")!=1 or len(cases)!=1 or p.get("checkpoint",{}).get("sha256")!=sha
    or not p.get("direct_compact_probabilities") or p.get("public_labels_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
name=cases[0]
if not (pred/name).is_file() or not (pred/f"{name.removesuffix('.nii.gz')}.prob16.npz").is_file():raise SystemExit(1)
PY
      then valid=1;fi
      [[ "${valid}" -eq 1 ]] && continue
      while :; do
        used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
        [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1536 ]] && break
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
checkpoint=Path(sys.argv[1]);checkpoint_sha=sys.argv[2];analysis=Path(sys.argv[3]);epoch=int(sys.argv[4]);output=Path(sys.argv[5]);payload=json.load(open(analysis))
if (payload.get("status")!="COMPLETE" or payload.get("case_count")!=50
    or payload.get("public_labels_read") is not True or payload.get("hidden_read")
    or payload.get("submission_created") or payload.get("uploaded")
    or len(payload.get("metrics",{}))!=4):raise SystemExit("invalid partial-band analysis")
record={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","epoch":epoch,"dataset_role":"public","case_count":50,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":checkpoint_sha},
 "analysis":{"path":str(analysis.resolve()),"sha256":hashlib.sha256(analysis.read_bytes()).hexdigest()},
 "public_labels_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(record,indent=2)+"\n");os.replace(temporary,output)
PY
  fi
  analysis_args+=(--analysis "${epoch}=${analysis}")
  checkpoint_args+=(--checkpoint "${epoch}=${checkpoint}")
done

"${python_bin}" -u "${selector}" --policy "${policy}" \
  "${analysis_args[@]}" "${checkpoint_args[@]}" --output "${terminal}"
trap - EXIT
