#!/usr/bin/env bash
set -euo pipefail

# Prepare only the 2,311 positive-only Train-Unlabel cases after the base S6
# trajectory and its serialized public evaluation have released host memory.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training_terminal="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827/TRAINING_TERMINAL.json"
public_terminal="${project_root}/nnunet/public_sweeps/resenc_s6_partialbox_endpoints_20260827/TERMINAL.json"
dataset_root="${project_root}/nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled"
build_marker="${dataset_root}/DATASET_BUILD_COMPLETE.json"
preprocess="${project_root}/nnunet/resume_preprocess.py"
output="${dataset_root}/PREPROCESSING_COMPLETE.json"
work="${project_root}/nnunet/runs/dataset502_preprocess_after_resenc_20260827"
# The projected cache for 2,311 new cases is 26.6 GiB and the audited 3x
# projection is 79.8 GiB. Preserve another 48 GiB for checkpoints, public
# probabilities, metadata, and filesystem variance before preprocessing.
minimum_disk_available_bytes=$((128 * 1024 * 1024 * 1024))

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "Dataset 502 preprocessing controller already active" >&2; exit 1; }

export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1

while :; do
  if [[ -s "${training_terminal}" && -s "${public_terminal}" ]]; then
    if "${python_bin}" - "${training_terminal}" "${public_terminal}" <<'PY'
import json,sys
training,public=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if (training.get("status")!="COMPLETE" or public.get("status")!="COMPLETE"
    or training.get("public_read") or training.get("hidden_read")
    or public.get("hidden_read") or public.get("submission_created")
    or public.get("uploaded")):
    raise SystemExit(1)
PY
    then break; fi
  fi
  printf '%s waiting for completed base ResEnc training and public terminals\n' "$(date -Ins)"
  sleep 30
done

for file in "${build_marker}" "${preprocess}"; do
  [[ -s "${file}" ]] || { echo "missing Dataset 502 preprocessing input: ${file}" >&2; exit 2; }
done
"${python_bin}" - "${build_marker}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("target_dataset")!="Dataset502_PanCancerPositiveUnlabeled"
    or p.get("labeled_cases")!=17575 or p.get("positive_unlabeled_cases")!=2311
    or p.get("total_cases")!=19886 or p.get("public_read") or p.get("hidden_read")):
    raise SystemExit("invalid Dataset 502 build marker")
PY

if [[ -s "${output}" ]]; then
  if "${python_bin}" - "${output}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if p.get("status")=="COMPLETE" and p.get("total_cases")==19886 else 1)
PY
  then exit 0; fi
  mv --no-clobber "${output}" "${work}/PREPROCESSING_COMPLETE.invalid_$(date +%Y%m%dT%H%M%S%N).json"
fi

while :; do
  read -r available_kib full_avg10 disk_available_bytes < <("${python_bin}" - "${dataset_root}" <<'PY'
import os,sys
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
filesystem=os.statvfs(sys.argv[1])
disk_available=filesystem.f_bavail*filesystem.f_frsize
print(memory["MemAvailable"],avg10,disk_available)
PY
  )
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" \
    "${disk_available_bytes}" "${minimum_disk_available_bytes}" <<'PY'
import sys
available,pressure,gpu0,gpu1,disk,minimum=(
    int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),
    int(sys.argv[5]),int(sys.argv[6])
)
raise SystemExit(0 if (available>=40*1024*1024 and pressure<=2.0
    and gpu0<=1024 and gpu1<=1024 and disk>=minimum) else 1)
PY
  then break; fi
  printf '%s waiting for safe Dataset 502 preprocessing: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB disk=%s minimum_disk=%s bytes\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}" \
    "${disk_available_bytes}" "${minimum_disk_available_bytes}"
  sleep 30
done

"${python_bin}" -u "${preprocess}" -d 502 -c 3d_fullres_ResEncS6 \
  -plans_name nnUNetPlans -np 2 >"${work}/preprocess.log" 2>&1

"${python_bin}" - "${dataset_root}" "${build_marker}" "${training_terminal}" \
  "${public_terminal}" "${preprocess}" "${output}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
dataset,build,training,public,script,output=map(Path,sys.argv[1:])
cache=dataset/"nnUNetPlans_3d_fullres_S4D2W32"
data={path.name.removesuffix(".b2nd") for path in cache.glob("*.b2nd") if not path.name.endswith("_seg.b2nd")}
seg={path.name.removesuffix("_seg.b2nd") for path in cache.glob("*_seg.b2nd")}
properties={path.name.removesuffix(".pkl") for path in cache.glob("*.pkl")}
if len(data)!=19886 or data!=seg or data!=properties:
    raise SystemExit("Dataset 502 preprocessed triplet inventory mismatch")
pseudo={name for name in data if name.startswith("Pseudo-")}
if len(pseudo)!=2311:
    raise SystemExit("Dataset 502 pseudo triplet inventory mismatch")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","dataset":"Dataset502_PanCancerPositiveUnlabeled",
 "configuration":"3d_fullres_ResEncS6","workers":2,
 "labeled_reused_cases":17575,"positive_unlabeled_preprocessed_cases":2311,
 "total_cases":19886,
 "build_marker":{"path":str(build.resolve()),"sha256":sha(build)},
 "base_training_terminal":{"path":str(training.resolve()),"sha256":sha(training)},
 "base_public_terminal":{"path":str(public.resolve()),"sha256":sha(public)},
 "preprocessor":{"path":str(script.resolve()),"sha256":sha(script)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
