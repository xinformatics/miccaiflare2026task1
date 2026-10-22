#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Same-run configurable STU-Net ensemble screen (epoch 20 by default). Wait for complete training and
# resume checkpoints, pause DDP, evaluate only the immutable public 50, and
# hand the decision to a separate label-blind saver. There is no hidden,
# archive, network, or upload operation in this controller.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stunet_root="${project_root}/external/STU-Net"
free_root="${project_root}/external/FreeTumor"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
interim_epoch="${STUNET_INTERIM_EPOCH:-20}"
[[ "${interim_epoch}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid interim epoch" >&2; exit 2; }
epoch4=$(printf '%04d' "${interim_epoch}")
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
fold_dir="${model_dir}/fold_all"
checkpoint="${fold_dir}/checkpoint_epoch_${epoch4}.pth"
resume_checkpoint="${fold_dir}/checkpoint_latest.pth"
training_unit="${STUNET_TRAINING_UNIT:-miccai-stunetb-training-after-independent-20260825.service}"
resume_unit="${STUNET_FAILURE_RESUME_UNIT:-miccai-stunetb-training-resume-e${interim_epoch}-20260825}"
training_script="${stunet_root}/pancancer/run_training_after_independent_audit_20260825.sh"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${gf_root}/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${free_root}/public_sweeps/independent_baseline_20260825/predictions"
output="${STUNET_INTERIM_OUTPUT:-${stunet_root}/pancancer/public_sweeps/stunetb_e${interim_epoch}_ensemble_20260825}"
predictions="${output}/predictions"
analysis="${output}/ensemble.json"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
pause_marker="${output}/TRAINING_PAUSED.json"
lock="${output}/controller.lock"
idle_memory_mib=4096
analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"
[[ "${analysis_workers}" =~ ^[1-9][0-9]*$ && "${analysis_workers}" -le 4 ]] || {
  echo "STUNET_ANALYSIS_WORKERS must be 1-4 for memory-safe full-volume scoring" >&2
  exit 2
}

mkdir -p "${output}" "${predictions}"
exec 9>"${lock}"
flock -n 9 || { echo "STU-Net epoch-${interim_epoch} public controller already active" >&2; exit 1; }

training_was_paused=0
resume_training() {
  [[ "${training_was_paused}" -eq 1 ]] || return 0
  if systemctl --user is-active --quiet "${resume_unit}.service"; then return 0; fi
  systemd-run --user --unit="${resume_unit}" --collect \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
    /usr/bin/bash "${training_script}"
}

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${interim_epoch}" "${checkpoint}" "${resume_checkpoint}" \
    "${pause_marker}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
reason=sys.argv[1];epoch=int(sys.argv[2]);checkpoint,resume,pause,summary,output=map(Path,sys.argv[3:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_TOUCHCOMP","reason":reason,
 "predeclared_epochs":[epoch],"selected_epoch":None,"selected_configuration":None,
 "checkpoint":asset(checkpoint),"resume_checkpoint":asset(resume),
 "training_pause":asset(pause),"summary":asset(summary),
 "public_read":any(output.parent.rglob("*.nii.gz")),"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "STU-Net epoch-${interim_epoch} public controller failed at line ${line}, status ${status}"
  fi
  if [[ "${status}" -ne 0 ]]; then resume_training || true; fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE) trap - EXIT;exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected epoch-${interim_epoch} public terminal status: ${existing}" >&2;exit 2 ;;
  esac
fi

# The checkpoint-specific and nnU-Net resume files must both be stable and
# describe exactly the requested number of completed epochs before DDP can be stopped.
stable=0
old_pair=""
while :; do
  if [[ -s "${checkpoint}" && -s "${resume_checkpoint}" ]]; then
    pair="$(stat -c '%s:%Y' "${checkpoint}"):$(stat -c '%s:%Y' "${resume_checkpoint}")"
    if [[ "${pair}" == "${old_pair}" ]]; then stable=$((stable+1));else stable=0;old_pair="${pair}";fi
    if [[ "${stable}" -ge 3 ]] && "${python_bin}" - "${interim_epoch}" "${checkpoint}" "${resume_checkpoint}" <<'PY'
import sys,numpy as np,torch
epoch=int(sys.argv[1])
allowed=[np._core.multiarray.scalar,np.dtype,type(np.dtype(np.float64)),type(np.dtype(np.float32)),type(np.dtype(np.int64)),type(np.dtype(np.bool_))]
with torch.serialization.safe_globals(allowed):
    for path in sys.argv[2:]:
        x=torch.load(path,map_location="cpu",weights_only=True)
        if (not isinstance(x,dict) or int(x.get("current_epoch",-1))!=epoch
            or not x.get("network_weights") or not x.get("optimizer_state") or not x.get("logging")):
            raise SystemExit(f"invalid epoch-{epoch} resume contract: {path}")
PY
    then break;fi
  fi
  printf '%s waiting for stable STU-Net epoch-%s checkpoints\n' "$(date -Ins)" "${interim_epoch}"
  sleep 15
done

checkpoint_sha_before=$(sha256sum "${checkpoint}" | awk '{print $1}')
resume_sha_before=$(sha256sum "${resume_checkpoint}" | awk '{print $1}')
if systemctl --user is-active --quiet "${training_unit}"; then systemctl --user stop "${training_unit}";fi
training_was_paused=1
while systemctl --user is-active --quiet "${training_unit}"; do sleep 2;done
[[ "${checkpoint_sha_before}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]
[[ "${resume_sha_before}" == "$(sha256sum "${resume_checkpoint}" | awk '{print $1}')" ]]
"${python_bin}" - "${checkpoint}" "${checkpoint_sha_before}" "${resume_checkpoint}" \
  "${resume_sha_before}" "${interim_epoch}" "${pause_marker}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
checkpoint,csha,resume,rsha,epoch,output=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]),sys.argv[4],int(sys.argv[5]),Path(sys.argv[6])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "reason":f"predeclared STU-Net epoch-{epoch} public ensemble screen",
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":csha},
 "resume_checkpoint":{"path":str(resume.resolve()),"sha256":rsha},
 "hidden_read":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${idle_memory_mib}" && "${used1}" -le "${idle_memory_mib}" ]]; then break;fi
  printf '%s waiting for epoch-%s public GPUs: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${interim_epoch}" "${used0}" "${used1}"
  sleep 15
done

for directory in "${images}" "${labels}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || { echo "invalid public inventory: ${directory}" >&2;exit 2; }
done

# Full-volume probability export has a much larger host-memory footprint than
# mask-only inference. Long-lived 25-case shards accumulated background export
# arrays and systemd-oomd killed the epoch-100 controller. Even a bounded pair
# crossed the host's pressure threshold with saturated swap, so run strictly
# one CT per process at a time and alternate GPUs. Valid interrupted outputs
# are reused, and process exit releases all case-specific memory.
for part_id in $(seq 0 49); do
  gpu=$((part_id % 2))
  part4=$(printf '%04d' "${part_id}")
  "${python_bin}" -u "${stunet_root}/pancancer/predict_stunet_safe.py" \
    --images "${images}" --output "${predictions}" --model-dir "${model_dir}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
    --part-id "${part_id}" --num-parts 50 --workers 1 --save-probabilities --dataset-role public \
    --terminal "${output}/CASE_${part4}.json" > "${output}/case_${part4}.log" 2>&1
done
[[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${predictions}" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${stunet_root}/pancancer/analyze_stunet_gf_ensemble.py" \
  --labels "${labels}" --anchor "${anchor}" --stunet "${predictions}" \
  --freetumor "${freetumor}" --stunet-probabilities "${predictions}" \
  --output "${analysis}" --workers "${analysis_workers}" \
  > "${output}/ensemble.log" 2>&1

"${python_bin}" - "${checkpoint}" "${resume_checkpoint}" "${pause_marker}" \
  "${analysis}" "${summary}" "${interim_epoch}" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
checkpoint,resume,pause,analysis_path,summary=map(Path,sys.argv[1:6]);epoch=int(sys.argv[6]);output=Path(sys.argv[7]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
analysis=json.load(open(analysis_path,encoding="utf-8"))
if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
    or analysis.get("role")!="fixed_public_ensemble_selection_only"
    or analysis.get("hidden_read") or analysis.get("submission_created") or analysis.get("uploaded")):
    raise SystemExit(f"invalid epoch-{epoch} public ensemble analysis")
rows=[{"checkpoint_epoch":epoch,**row} for row in analysis["deployments"]]
rows.sort(key=lambda row:(row["Lesion_DSC"],row["Lesion_NSD"]),reverse=True)
eligible=[row for row in rows if row["Lesion_DSC"]>=0.6154316379500749 and row["Lesion_NSD"]>=0.5104225107769019]
best=eligible[0] if eligible else rows[0];gate=bool(eligible)
with open(summary,"w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=["checkpoint_epoch","configuration","Lesion_DSC","Lesion_NSD","DSC_gain","NSD_gain"],delimiter="\t")
    writer.writeheader();writer.writerows(rows)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_SUBMISSION_PREPARATION" if gate else "KEEP_TOUCHCOMP",
 "predeclared_epochs":[epoch],"selection_policy":"maximum fixed-public DSC subject to immutable DSC/NSD gates",
 "selected_epoch":epoch if gate else None,"selected_configuration":best["configuration"] if gate else None,
 "selected_fixed_public_DSC":best["Lesion_DSC"] if gate else None,
 "selected_fixed_public_NSD":best["Lesion_NSD"] if gate else None,"best_observed":best,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)} if gate else None,
 "selected_analysis":{"path":str(analysis_path.resolve()),"sha256":sha(analysis_path)} if gate else None,
 "children":[{"epoch":epoch,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
              "analysis":{"path":str(analysis_path.resolve()),"sha256":sha(analysis_path)}}],
 "resume_checkpoint":{"path":str(resume.resolve()),"sha256":sha(resume)},
 "training_pause":{"path":str(pause.resolve()),"sha256":sha(pause)},
 "summary":{"path":str(summary.resolve()),"sha256":sha(summary)},
 "fixed_public_DSC_gate":0.6154316379500749,"fixed_public_NSD_floor":0.5104225107769019,
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

# Successful public completion deliberately leaves DDP paused. The save-only
# consumer makes the hidden/no-hidden decision, then a separate controller
# resumes this exact run on every final saver state.
trap - EXIT
cat "${summary}"
