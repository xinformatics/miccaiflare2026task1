#!/usr/bin/env bash
set -euo pipefail

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
tri_root="${gf}/docker_submission/analysis/three_route_shared_focus_public_equivalence_20260827"
tri_terminal="${tri_root}/EQUIVALENCE.json"
tri_lock="${tri_root}/controller.lock"
bbox="${tri_root}/full/bbox"
guided="${tri_root}/full/guided"
cbg="${tri_root}/full/cbg"
whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
reference="${gf}/analysis/lesion_error_touchcomp_20260825/predictions"
materializer="${gf}/docker_submission/materialize_runtime_anchor.py"
output="${gf}/docker_submission/analysis/runtime_anchor_equivalence_20260827"
predictions="${output}/predictions"
materialize_terminal="${output}/MATERIALIZE.json"
terminal="${output}/EQUIVALENCE.json"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "runtime-anchor equivalence already active" >&2; exit 1; }
while [[ ! -s "${tri_terminal}" ]]; do
  printf '%s waiting for three-route shared-Focus terminal\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${tri_lock}"
flock 8
"${python_bin}" - "${tri_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_raw_route_equivalence")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid three-route equivalence prerequisite")
PY
for directory in "${bbox}" "${whole}" "${tumor}" "${guided}" "${cbg}" "${reference}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "runtime-anchor inventory is not 50: ${directory}" >&2; exit 2;
  }
done
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
available,pressure,used0,used1=int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
raise SystemExit(0 if available>=30*1024*1024 and pressure<=2.0 and max(used0,used1)<=1024 else 1)
PY
  then break; fi
  printf '%s waiting for safe runtime-anchor host: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
  sleep 30
done
nice -n 15 ionice -c 3 "${python_bin}" -u "${materializer}" \
  --bbox "${bbox}" --prompt-whole "${whole}" --prompt-tumor "${tumor}" \
  --guided "${guided}" --cbg "${cbg}" --output "${predictions}" \
  --terminal "${materialize_terminal}" --workers 1 --dataset-role public \
  >"${output}/materialize.log" 2>&1

"${python_bin}" - "${reference}" "${predictions}" "${materialize_terminal}" \
  "${tri_terminal}" "${materializer}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
reference,predictions,materialized,tri,script,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.load(open(materialized,encoding="utf-8"));names=sorted(p.name for p in reference.glob("*.nii.gz"))
if (len(names)!=50 or m.get("status")!="COMPLETE" or m.get("case_count")!=50
    or m.get("public_labels_read") or m.get("hidden_read")
    or m.get("submission_created") or m.get("uploaded")):
    raise SystemExit("invalid runtime-anchor materialization terminal")
mismatches=[];geometry=[]
for name in names:
    r=nib.load(str(reference/name));o=nib.load(str(predictions/name))
    if not (r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6)):geometry.append(name)
    if not np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)):mismatches.append(name)
exact=not mismatches and not geometry
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_RUNTIME_ANCHOR" if exact else "REJECT_RUNTIME_ANCHOR",
 "deployment_authorized":False,"case_count":50,"exact_anchor_equivalence":exact,
 "mismatches":mismatches,"geometry_failures":geometry,
 "materializer":{"path":str(script.resolve()),"sha256":sha(script)},
 "materialization":{"path":str(materialized.resolve()),"sha256":sha(materialized),"wall_seconds":m["wall_seconds"]},
 "three_route":{"path":str(tri.resolve()),"sha256":sha(tri)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps(payload,indent=2))
if not exact:raise SystemExit("runtime anchor failed exact public equivalence")
PY
