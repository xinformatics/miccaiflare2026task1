#!/usr/bin/env bash
set -euo pipefail

# Public-only, two-GPU evaluation of the published FreeTumor abdominal model.
# This is a proposal-complementarity diagnostic, not a submission builder. It
# waits behind the complete GF pipeline and cannot read hidden inputs, create a
# ZIP, or upload anything.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
free_root="${project_root}/external/FreeTumor"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream="${gf_root}/public_sweeps/gf_focus_pu_e303_interim_saved_20260825/TERMINAL.json"
images="${project_root}/validation/Validation-Public-Images"
labels="${project_root}/validation/Validation-Public-Labels"
anchor="${gf_root}/analysis/lesion_error_touchcomp_20260825/predictions"
checkpoint="${free_root}/pretrained_models/baseline/model_baseline_segmentor.pt"
output="${free_root}/public_sweeps/independent_baseline_20260825"
predictions="${output}/predictions"
terminal="${output}/TERMINAL.json"
lock="${output}/controller.lock"
expected_checkpoint_sha="c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
idle_memory_mib=4096

mkdir -p "${output}" "${predictions}"
exec 9>"${lock}"
flock -n 9 || { echo "FreeTumor public diagnostic already active" >&2; exit 1; }

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${checkpoint}" "${terminal}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
reason=sys.argv[1];checkpoint,output=map(Path,sys.argv[2:])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","reason":reason,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()} if checkpoint.is_file() else None,
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
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
    "FreeTumor public diagnostic failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) trap - EXIT; exit 0 ;;
    FAILED)
      preserved="${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved}"
      ;;
    *) echo "unexpected FreeTumor terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

observed_checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
[[ "${observed_checkpoint_sha}" == "${expected_checkpoint_sha}" ]]

while :; do
  if [[ ! -s "${upstream}" ]]; then
    printf '%s waiting for checkpoint-303 saver terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  upstream_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${upstream}")
  case "${upstream_status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed checkpoint-303 saver terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected recall-shortlist saver status: ${upstream_status}" >&2; exit 2 ;;
  esac
done

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${idle_memory_mib}" \
      && "${used1}" -le "${idle_memory_mib}" ]]; then
    break
  fi
  printf '%s waiting for idle GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

for shard in 0 1; do
  "${python_bin}" -u "${free_root}/evaluate_public_safe.py" \
    --images "${images}" --output "${predictions}" \
    --checkpoint "${checkpoint}" --gpu "${shard}" \
    --shard-index "${shard}" --num-shards 2 --workers 2 \
    --sw-batch-size 4 --overlap 0.75 \
    --terminal "${output}/SHARD_${shard}.json" \
    > "${output}/shard_${shard}.log" 2>&1 &
  shard_pids[${shard}]=$!
done

status=0
wait "${shard_pids[0]}" || status=$?
wait "${shard_pids[1]}" || status=$?
[[ "${status}" -eq 0 ]]

"${python_bin}" - "${images}" "${labels}" "${predictions}" \
  "${checkpoint}" "${expected_checkpoint_sha}" \
  "${output}/SHARD_0.json" "${output}/SHARD_1.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,labels,predictions,checkpoint=map(Path,sys.argv[1:5]);expected=sys.argv[5]
terminals=list(map(Path,sys.argv[6:]))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if sha(checkpoint)!=expected: raise SystemExit("checkpoint changed during inference")
image_names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")}
label_names={p.name for p in labels.glob("*.nii.gz")}
prediction_names={p.name for p in predictions.glob("*.nii.gz")}
if len(image_names)!=50 or image_names!=label_names or image_names!=prediction_names:
    raise SystemExit("public inventory mismatch")
terminal_cases=set()
for index,path in enumerate(terminals):
    payload=json.load(open(path,encoding="utf-8"))
    if (payload.get("status")!="COMPLETE" or payload.get("shard_index")!=index
        or payload.get("num_shards")!=2 or payload.get("checkpoint",{}).get("sha256")!=expected
        or payload.get("public_labels_read") or payload.get("hidden_read")
        or payload.get("submission_created") or payload.get("uploaded")):
        raise SystemExit(f"invalid shard terminal {index}")
    terminal_cases.update(payload.get("cases",[]))
if terminal_cases!=image_names: raise SystemExit("shard case coverage mismatch")
for name in sorted(image_names):
    reference=nib.load(str(images/name.replace(".nii.gz","_0000.nii.gz")))
    prediction=nib.load(str(predictions/name))
    array=np.asanyarray(prediction.dataobj)
    if prediction.shape!=reference.shape or not np.allclose(prediction.affine,reference.affine,atol=1e-4):
        raise SystemExit(f"geometry mismatch: {name}")
    if not np.isin(np.unique(array),[0,1]).all(): raise SystemExit(f"nonbinary: {name}")
PY

MPLCONFIGDIR="${output}/.matplotlib" \
  "${python_bin}" -u "${gf_root}/evaluate_public_parallel.py" \
    --pred "${predictions}" --gt "${labels}" \
    --csv "${output}/standalone.metrics.csv" --workers 8 \
    > "${output}/standalone.score.log"

"${python_bin}" -u "${free_root}/analyze_gf_complementarity.py" \
  --labels "${labels}" --anchor "${anchor}" --candidate "${predictions}" \
  --output "${output}/complementarity.json" \
  > "${output}/complementarity.log"

"${python_bin}" - "${upstream}" "${checkpoint}" "${predictions}" \
  "${output}/standalone.metrics.csv" "${output}/complementarity.json" \
  "${output}/SHARD_0.json" "${output}/SHARD_1.json" "${terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
upstream,checkpoint,predictions,metrics,analysis,shard0,shard1,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader(open(metrics,encoding="utf-8")))
if len(rows)!=50: raise SystemExit("standalone metric inventory is not 50")
complement=json.load(open(analysis,encoding="utf-8"))
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"PUBLIC_DIAGNOSTIC_ONLY",
 "model":"published FreeTumor abdominal baseline SwinUNETR",
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "upstream":{"path":str(upstream.resolve()),"sha256":sha(upstream)},
 "predictions":{"path":str(predictions.resolve()),"case_count":50},
 "standalone_metrics":{"path":str(metrics.resolve()),"sha256":sha(metrics),
   "Lesion_DSC":sum(float(r["Lesion_DSC"]) for r in rows)/len(rows),
   "Lesion_NSD":sum(float(r["Lesion_NSD"]) for r in rows)/len(rows)},
 "complementarity":{"path":str(analysis.resolve()),"sha256":sha(analysis),
   "large_misses_recovered":complement["large_misses_recovered_by_candidate"],
   "closed_gf_cases_recovered":complement["closed_gf_cases_recovered_by_candidate"],
   "best_direct_union":complement["best_direct_union"],
   "direct_union_public_gate":complement["direct_union_public_gate"]},
 "shards":[{"path":str(p.resolve()),"sha256":sha(p)} for p in (shard0,shard1)],
 "public_images_read":True,"public_labels_read":True,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

trap - EXIT
cat "${terminal}"
