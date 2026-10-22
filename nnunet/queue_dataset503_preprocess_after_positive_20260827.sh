#!/usr/bin/env bash
set -euo pipefail

# Process only the four new organ-target cases after the standard positive
# branch and its serialized public evaluation have released host/GPU memory.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
positive_public="${project_root}/nnunet/public_sweeps/resenc_s6_positive_unlabeled_endpoints_20260827/TERMINAL.json"
dataset_root="${project_root}/nnUNet_preprocessed/Dataset503_PanCancerOrganTarget"
build_marker="${dataset_root}/DATASET_BUILD_COMPLETE.json"
preprocess="${project_root}/nnunet/resume_preprocess.py"
output="${dataset_root}/PREPROCESSING_COMPLETE.json"
work="${project_root}/nnunet/runs/dataset503_preprocess_after_positive_20260827"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "Dataset503 preprocessing controller already active" >&2; exit 1; }

export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1

while [[ ! -s "${positive_public}" ]]; do
  printf '%s waiting for completed positive-unlabeled public selector\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${positive_public}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not isinstance(p.get("diagnostic_best"),dict)
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("positive public selector is not a safe complete parent")
PY

for file in "${build_marker}" "${preprocess}"; do
  [[ -s "${file}" ]] || { echo "missing Dataset503 preprocessing input: ${file}" >&2; exit 2; }
done
"${python_bin}" - "${build_marker}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE"
    or p.get("target_dataset")!="Dataset503_PanCancerOrganTarget"
    or p.get("labeled_cases")!=17575 or p.get("organ_target_cases")!=4
    or p.get("organ_target_components")!=5 or p.get("total_cases")!=17579
    or p.get("public_read") or p.get("hidden_read")):
    raise SystemExit("invalid Dataset503 build marker")
PY

if [[ -s "${output}" ]]; then
  if "${python_bin}" - "${output}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if p.get("status")=="COMPLETE" and p.get("total_cases")==17579 else 1)
PY
  then exit 0; fi
  mv --no-clobber "${output}" "${work}/PREPROCESSING_COMPLETE.invalid_$(date +%Y%m%dT%H%M%S%N).json"
fi

while :; do
  read -r available_kib full_avg10 < <("${python_bin}" - <<'PY'
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
print(memory["MemAvailable"],avg10)
PY
  )
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used0}" "${used1}" <<'PY'
import sys
available,pressure,gpu0,gpu1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if available>=40*1024*1024 and pressure<=2.0 and gpu0<=1024 and gpu1<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe Dataset503 preprocessing: available=%s KiB pressure=%s gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done

"${python_bin}" -u "${preprocess}" -d 503 -c 3d_fullres_ResEncS6 \
  -plans_name nnUNetPlans -np 2 >"${work}/preprocess.log" 2>&1

"${python_bin}" - "${dataset_root}" "${build_marker}" "${positive_public}" \
  "${preprocess}" "${output}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
dataset,build,positive,script,output=map(Path,sys.argv[1:])
cache=dataset/"nnUNetPlans_3d_fullres_S4D2W32"
data={p.name.removesuffix(".b2nd") for p in cache.glob("*.b2nd") if not p.name.endswith("_seg.b2nd")}
seg={p.name.removesuffix("_seg.b2nd") for p in cache.glob("*_seg.b2nd")}
properties={p.name.removesuffix(".pkl") for p in cache.glob("*.pkl")}
if len(data)!=17579 or data!=seg or data!=properties:
    raise SystemExit("Dataset503 preprocessed triplet inventory mismatch")
target={name for name in data if name.startswith("PseudoOrgan-")}
if target!={f"PseudoOrgan-AMOS22_{i:04d}" for i in range(1,5)}:
    raise SystemExit("Dataset503 target triplet inventory mismatch")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","dataset":"Dataset503_PanCancerOrganTarget",
 "configuration":"3d_fullres_ResEncS6","workers":2,
 "labeled_reused_cases":17575,"organ_target_preprocessed_cases":4,
 "total_cases":17579,
 "build_marker":{"path":str(build.resolve()),"sha256":sha(build)},
 "positive_public_selector":{"path":str(positive.resolve()),"sha256":sha(positive)},
 "preprocessor":{"path":str(script.resolve()),"sha256":sha(script)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
