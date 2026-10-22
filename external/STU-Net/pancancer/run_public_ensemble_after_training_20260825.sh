#!/usr/bin/env bash
set -euo pipefail

# Public-only trajectory screen for the independent STU-Net-B proposal model.
# Checkpoints 100/200/300 and the component ensemble grid were predeclared
# before inference. Hidden data, packaging, networking, and upload are absent.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stunet_root="${project_root}/external/STU-Net"
free_root="${project_root}/external/FreeTumor"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${stunet_root}/pancancer/runs/stunetb_partialbox_300ep_20260825/TRAINING_TERMINAL.json"
freetumor_terminal="${free_root}/public_sweeps/independent_baseline_20260825/TERMINAL.json"
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${gf_root}/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${free_root}/public_sweeps/independent_baseline_20260825/predictions"
output="${stunet_root}/pancancer/public_sweeps/stunetb_ensemble_trajectory_20260825"
terminal="${output}/TERMINAL.json"
summary="${output}/summary.tsv"
lock="${output}/controller.lock"
epochs=(100 200 300)
idle_memory_mib=4096
analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"
compact_equivalence="${stunet_root}/pancancer/public_sweeps/direct_compact_equivalence_e100_20260826/EQUIVALENCE.json"
[[ "${analysis_workers}" =~ ^[1-9][0-9]*$ && "${analysis_workers}" -le 4 ]] || {
  echo "STUNET_ANALYSIS_WORKERS must be 1-4 for memory-safe full-volume scoring" >&2
  exit 2
}

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "STU-Net public ensemble controller already active" >&2; exit 1; }

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${training}" "${freetumor_terminal}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
reason=sys.argv[1];training,freetumor,summary,output=map(Path,sys.argv[2:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_TOUCHCOMP","reason":reason,
 "training_terminal":asset(training),"freetumor_terminal":asset(freetumor),"summary":asset(summary),
 "predeclared_epochs":[100,200,300],"selected_epoch":None,"selected_configuration":None,
 "public_read":any(output.parent.rglob("*.nii.gz")),"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure_line=unknown
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_failure \
    "STU-Net public ensemble controller failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing_status}" in
    COMPLETE) trap - EXIT;exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected STU-Net public terminal status: ${existing_status}" >&2;exit 2 ;;
  esac
fi

wait_for_complete() {
  local path="$1" description="$2"
  while :; do
    if [[ ! -s "${path}" ]]; then printf '%s waiting for %s\n' "$(date -Ins)" "${description}";sleep 30;continue;fi
    status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${path}")
    case "${status}" in
      COMPLETE) break ;;
      FAILED) printf '%s waiting for replacement of failed %s\n' "$(date -Ins)" "${description}";sleep 30 ;;
      *) echo "unexpected ${description} status: ${status}" >&2;exit 2 ;;
    esac
  done
}
wait_for_complete "${training}" "STU-Net training terminal"
wait_for_complete "${freetumor_terminal}" "FreeTumor public terminal"

"${python_bin}" - "${training}" "${freetumor_terminal}" "${model_dir}" <<'PY'
import hashlib,json,sys
from pathlib import Path
training_path,free_path,model_dir=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
training=json.load(open(training_path,encoding="utf-8"));free=json.load(open(free_path,encoding="utf-8"))
if (training.get("status")!="COMPLETE" or training.get("public_read") or training.get("hidden_read")
    or training.get("submission_created") or training.get("uploaded")):
    raise SystemExit("invalid STU-Net training terminal")
if (free.get("status")!="COMPLETE" or free.get("decision")!="PUBLIC_DIAGNOSTIC_ONLY"
    or free.get("hidden_read") or free.get("submission_created") or free.get("uploaded")):
    raise SystemExit("invalid FreeTumor public terminal")
trajectory={Path(item["path"]).resolve():item["sha256"] for item in training.get("trajectory_checkpoints",[])}
for epoch in (100,200,300):
    checkpoint=model_dir/"fold_all"/f"checkpoint_epoch_{epoch:04d}.pth"
    if not checkpoint.is_file() or trajectory.get(checkpoint.resolve())!=sha(checkpoint):
        raise SystemExit(f"missing hash-bound STU-Net checkpoint {epoch}")
PY
for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected exact public 50: ${directory}" >&2;exit 2;
  }
done

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${idle_memory_mib}" && "${used1}" -le "${idle_memory_mib}" ]]; then break;fi
  printf '%s waiting for STU-Net public GPUs: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}";sleep 30
done

compact_args=()
source_checkpoint="${model_dir}/fold_all/checkpoint_epoch_0100.pth"
if [[ -s "${compact_equivalence}" ]] && "${python_bin}" - "${compact_equivalence}" "${source_checkpoint}" <<'PY'
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

infer_epoch_casewise() {
  local epoch="$1" epoch_dir checkpoint gpu part_id part4
  epoch_dir=$(printf '%s/epoch_%04d' "${output}" "${epoch}")
  checkpoint=$(printf 'checkpoint_epoch_%04d.pth' "${epoch}")
  mkdir -p "${epoch_dir}/predictions"
  for part_id in $(seq 0 49); do
    gpu=$((part_id % 2));part4=$(printf '%04d' "${part_id}")
    "${python_bin}" -u "${stunet_root}/pancancer/predict_stunet_safe.py" \
      --images "${images}" --output "${epoch_dir}/predictions" --model-dir "${model_dir}" \
      --checkpoint "${checkpoint}" --gpu "${gpu}" --part-id "${part_id}" --num-parts 50 \
      --workers 1 --save-probabilities "${compact_args[@]}" --dataset-role public \
      --terminal "${epoch_dir}/CASE_${part4}.json" > "${epoch_dir}/case_${part4}.log" 2>&1
  done
}

for epoch in "${epochs[@]}"; do infer_epoch_casewise "${epoch}";done

for epoch in "${epochs[@]}"; do
  epoch_dir=$(printf '%s/epoch_%04d' "${output}" "${epoch}")
  [[ $(find "${epoch_dir}/predictions" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  [[ $(find "${epoch_dir}/predictions" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${stunet_root}/pancancer/analyze_stunet_gf_ensemble.py" \
    --labels "${labels}" --anchor "${anchor}" --stunet "${epoch_dir}/predictions" \
    --freetumor "${freetumor}" --stunet-probabilities "${epoch_dir}/predictions" \
    --output "${epoch_dir}/ensemble.json" --workers "${analysis_workers}" \
    > "${epoch_dir}/ensemble.log" 2>&1
done

"${python_bin}" - "${training}" "${freetumor_terminal}" "${model_dir}" \
  "${output}" "${summary}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
training,free,model_dir,root,summary,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];children=[]
for epoch in (100,200,300):
    analysis_path=root/f"epoch_{epoch:04d}"/"ensemble.json";analysis=json.load(open(analysis_path,encoding="utf-8"))
    if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
        or analysis.get("role")!="fixed_public_ensemble_selection_only"
        or analysis.get("hidden_read") or analysis.get("submission_created") or analysis.get("uploaded")):
        raise SystemExit(f"invalid STU-Net public ensemble analysis {epoch}")
    checkpoint=model_dir/"fold_all"/f"checkpoint_epoch_{epoch:04d}.pth"
    children.append({"epoch":epoch,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
                     "analysis":{"path":str(analysis_path.resolve()),"sha256":sha(analysis_path)}})
    for row in analysis["deployments"]: rows.append({"checkpoint_epoch":epoch,**row})
rows.sort(key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]),reverse=True)
eligible=[row for row in rows if row["Lesion_DSC"]>=0.6154316379500749 and row["Lesion_NSD"]>=0.5104225107769019]
best=eligible[0] if eligible else rows[0];gate=bool(eligible)
with open(summary,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=["checkpoint_epoch","configuration","Lesion_DSC","Lesion_NSD","DSC_gain","NSD_gain"],delimiter="\t")
    writer.writeheader();writer.writerows(rows)
selected=next(item for item in children if item["epoch"]==best["checkpoint_epoch"])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION" if gate else "KEEP_TOUCHCOMP",
 "predeclared_epochs":[100,200,300],"selection_policy":"maximum fixed-public DSC subject to immutable DSC/NSD gates",
 "selected_epoch":best["checkpoint_epoch"] if gate else None,
 "selected_configuration":best["configuration"] if gate else None,
 "selected_fixed_public_DSC":best["Lesion_DSC"] if gate else None,
 "selected_fixed_public_NSD":best["Lesion_NSD"] if gate else None,
 "best_observed":best,"checkpoint":selected["checkpoint"] if gate else None,
 "selected_analysis":selected["analysis"] if gate else None,
 "children":children,"training_terminal":{"path":str(training.resolve()),"sha256":sha(training)},
 "freetumor_terminal":{"path":str(free.resolve()),"sha256":sha(free)},
 "summary":{"path":str(summary.resolve()),"sha256":sha(summary)},
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
trap - EXIT
cat "${summary}"
