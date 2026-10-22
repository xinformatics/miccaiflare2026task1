#!/usr/bin/env bash
set -euo pipefail

# All-50 public-label-free equivalence/runtime gate for the FreeTumor-free
# existing-support proxy and sparse STU p70 confirmer.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
gf="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${gf}/analysis/lesion_error_touchcomp_20260825/predictions"
bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
whole="${project_root}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project_root}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
full_stu="${root}/public_sweeps/stunetb_matched_e160_20260826/component_predictions"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
predictor="${root}/predict_stunet_proxy_sparse.py"
materializer="${gf}/docker_submission/materialize_runtime_proxy_reference.py"
probe_root="${root}/public_sweeps/stunet_proxy_sparse_probe_20260827"
probe_terminal="${probe_root}/EQUIVALENCE.json"
probe_lock="${probe_root}/controller.lock"
output="${root}/public_sweeps/stunet_proxy_sparse_all50_20260827"
reference="${output}/reference"
reference_terminal="${output}/REFERENCE.json"
predictions="${output}/predictions"
terminal="${output}/EQUIVALENCE.json"
gpu="${PROXY_SPARSE_ALL50_GPU:-0}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-proxy-sparse-all50
export PYTHONPATH="${gf}/docker_submission:${gf}:${root}:${PYTHONPATH:-}"
[[ "${gpu}" =~ ^[01]$ ]] || { echo "PROXY_SPARSE_ALL50_GPU must be 0 or 1" >&2; exit 2; }
mkdir -p "${output}" "${reference}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "proxy-sparse all-50 controller already active" >&2; exit 1; }

wait_for_safe_host() {
  local stage="$1" available_kib full_avg10 used0 used1
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
    then return; fi
    printf '%s waiting for safe host before %s: available=%s KiB pressure=%s gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${stage}" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
    sleep 30
  done
}

while [[ ! -s "${probe_terminal}" ]]; do
  printf '%s waiting for proxy-sparse stress probe terminal\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${probe_lock}"
flock 8
probe_pass=$("${python_bin}" - "${probe_terminal}" "${checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"));checkpoint=Path(sys.argv[2])
sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
valid=(p.get("status")=="COMPLETE" and p.get("probe_pass")
       and p.get("mask_equal") and p.get("geometry_equal")
       and p.get("runtime_eligible_20s")
       and p.get("checkpoint",{}).get("sha256")==sha
       and not p.get("deployment_authorized") and not p.get("public_labels_read")
       and not p.get("hidden_read") and not p.get("submission_created") and not p.get("uploaded"))
print(str(bool(valid)).lower())
PY
)
if [[ "${probe_pass}" != true ]]; then
  "${python_bin}" - "${probe_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
probe,output=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"PROXY_SPARSE_STRESS_PROBE_FAILED",
 "deployment_authorized":False,"probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "public_labels_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
  exit 0
fi

for directory in "${images}" "${anchor}" "${bbox}" "${whole}" "${tumor}" "${full_stu}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "proxy-sparse all-50 inventory is not 50: ${directory}" >&2; exit 2;
  }
done
for file in "${checkpoint}" "${predictor}" "${materializer}"; do
  [[ -s "${file}" ]] || { echo "missing proxy-sparse all-50 input: ${file}" >&2; exit 2; }
done

reference_ok=false
if [[ -s "${reference_terminal}" ]]; then
  reference_ok=$("${python_bin}" - "${reference_terminal}" "${reference}" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"));root=Path(sys.argv[2])
print(str(bool(p.get("status")=="COMPLETE" and p.get("case_count")==50
               and p.get("dataset_role")=="public" and not p.get("public_labels_read")
               and not p.get("hidden_read") and not p.get("submission_created")
               and not p.get("uploaded") and len(list(root.glob("*.nii.gz")))==50)).lower())
PY
  )
fi
if [[ "${reference_ok}" != true ]]; then
  wait_for_safe_host "proxy reference materialization"
  nice -n 15 ionice -c 3 "${python_bin}" -u "${materializer}" \
    --anchor "${anchor}" --bbox "${bbox}" --prompt-whole "${whole}" \
    --prompt-tumor "${tumor}" --stunet "${full_stu}" \
    --output "${reference}" --terminal "${reference_terminal}" \
    --workers 1 --dataset-role public >"${output}/reference.log" 2>&1
fi

checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
for part_id in $(seq 0 49); do
  part4=$(printf '%04d' "${part_id}")
  case_terminal="${output}/CASE_${part4}.json"
  valid=0
  if [[ -s "${case_terminal}" ]] && "${python_bin}" - "${case_terminal}" "${checkpoint_sha}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"));rows=p.get("cases") or []
valid=(p.get("status")=="COMPLETE" and p.get("case_count")==1 and len(rows)==1
       and p.get("checkpoint",{}).get("sha256")==sys.argv[2]
       and p.get("dataset_role")=="public" and not p.get("public_labels_read")
       and not p.get("hidden_read") and not p.get("submission_created") and not p.get("uploaded"))
raise SystemExit(0 if valid else 1)
PY
  then valid=1; fi
  [[ "${valid}" -eq 1 ]] && continue
  wait_for_safe_host "proxy case ${part_id}"
  timeout --signal=TERM 120s "${python_bin}" -u "${predictor}" \
    --images "${images}" --anchors "${anchor}" --bbox-raw "${bbox}" \
    --prompt-whole "${whole}" --prompt-tumor "${tumor}" \
    --output "${predictions}" --model-dir "${model}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
    --part-id "${part_id}" --num-parts 50 --workers 1 \
    --dataset-role public --terminal "${case_terminal}" \
    >"${output}/case_${part4}.log" 2>&1
done

[[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]

"${python_bin}" - "${images}" "${reference}" "${predictions}" "${output}" \
  "${checkpoint}" "${probe_terminal}" "${reference_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,reference,predictions,root,checkpoint,probe,reference_terminal,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))
rows=[];mismatches=[];geometry=[]
for index,name in enumerate(names):
    case_terminal=root/f"CASE_{index:04d}.json";p=json.load(open(case_terminal,encoding="utf-8"));case_rows=p.get("cases") or []
    if (p.get("status")!="COMPLETE" or p.get("case_count")!=1 or len(case_rows)!=1
        or case_rows[0].get("case")!=name or p.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
        or p.get("public_labels_read") or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
        raise SystemExit(f"invalid proxy-sparse case terminal {index}")
    r=nib.load(str(reference/name));o=nib.load(str(predictions/name))
    equal=bool(np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)))
    geom=bool(r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6))
    if not equal:mismatches.append(name)
    if not geom:geometry.append(name)
    row=case_rows[0]
    rows.append({"case":name,"mask_equal":equal,"geometry_equal":geom,
      "wall_seconds":float(p["wall_seconds"]),"case_runtime_seconds":float(row["seconds"]),
      "stunet_skipped":bool(row["stunet_skipped"]),"full_tiles":int(row["full_tiles"]),
      "selected_tiles":int(row["selected_tiles"])})
times=sorted(row["wall_seconds"] for row in rows)
def q(value):
 position=(len(times)-1)*value;lo=int(math.floor(position));hi=int(math.ceil(position))
 return times[lo]+(times[hi]-times[lo])*(position-lo)
exact=not mismatches and not geometry;runtime=max(times)<=20.0;authorized=bool(exact and runtime)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"AUTHORIZE_PROXY_SPARSE_STAGE" if authorized else "REJECT_PROXY_SPARSE_STAGE",
 "deployment_authorized":authorized,"case_count":50,"exact_full_volume_proxy_equivalence":exact,
 "mismatched_cases":mismatches,"geometry_failures":geometry,"runtime_eligible_20s":runtime,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "stunet_skipped_cases":sum(row["stunet_skipped"] for row in rows),
 "full_tiles":sum(row["full_tiles"] for row in rows),"selected_tiles":sum(row["selected_tiles"] for row in rows),
 "cases":rows,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "reference":{"path":str(reference_terminal.resolve()),"sha256":sha(reference_terminal)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps({key:payload[key] for key in ("decision","exact_full_volume_proxy_equivalence","runtime_eligible_20s","runtime_seconds","stunet_skipped_cases","full_tiles","selected_tiles")},indent=2))
PY
