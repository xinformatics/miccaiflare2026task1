#!/usr/bin/env bash
set -euo pipefail

# Public-label-free proof that bbox, CBG, and Biomed-guided GF routes can share
# one Focus crop cache without changing any native-space output mask.

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
stunet="${project}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${gf}/docker_submission/predict_dual_glance_shared_focus.py"
images="${project}/validation/Validation-Public-Images"
proposal="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
reference_bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
reference_cbg="${gf}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/inference"
reference_guided="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
biomed_terminal="${gf}/docker_submission/analysis/biomedparse_dual_prompt_public_equivalence_20260827/EQUIVALENCE.json"
upstream_lock="${stunet}/public_sweeps/stunet_proxy_sparse_all50_20260827/controller.lock"
output="${gf}/docker_submission/analysis/three_route_shared_focus_public_equivalence_20260827"
probe_bbox="${output}/probe/bbox"
probe_cbg="${output}/probe/cbg"
probe_guided="${output}/probe/guided"
full_bbox="${output}/full/bbox"
full_cbg="${output}/full/cbg"
full_guided="${output}/full/guided"
probe_terminal="${output}/PROBE_PREDICT.json"
full_terminal="${output}/FULL_PREDICT.json"
terminal="${output}/EQUIVALENCE.json"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-three-route-focus
mkdir -p "${probe_bbox}" "${probe_cbg}" "${probe_guided}" \
  "${full_bbox}" "${full_cbg}" "${full_guided}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "three-route shared-Focus controller already active" >&2; exit 1; }

while [[ ! -s "${biomed_terminal}" ]]; do
  printf '%s waiting for persistent BiomedParse equivalence\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${biomed_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_prompt_outputs")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid persistent BiomedParse prerequisite")
PY

# The all-50 sparse-STU controller holds this lock until its complete replay or
# fail-closed skip is over. This prevents full-volume inference overlap.
exec 8>"${upstream_lock}"
flock 8

for directory in "${images}" "${proposal}" "${reference_bbox}" \
    "${reference_cbg}" "${reference_guided}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "three-route fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
[[ -s "${predictor}" ]]

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

wait_for_safe_host "three-route probe"
"${python_bin}" -u "${predictor}" --images "${images}" \
  --bbox-output "${probe_bbox}" --cbg-output "${probe_cbg}" \
  --guided-proposal "${proposal}" --guided-output "${probe_guided}" \
  --terminal "${probe_terminal}" --dataset-role public --gpu 0 \
  --part-id 14 --num-parts 50 --workers 0 --focus-batch-size 4 \
  >"${output}/probe.log" 2>&1

"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${reference_guided}" "${probe_bbox}" "${probe_cbg}" "${probe_guided}" \
  "${probe_terminal}" <<'PY'
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,rg,oa,ob,og,terminal=map(Path,sys.argv[1:])
name=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))[14]
p=json.load(open(terminal,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1 or not p.get("guided_route")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid three-route probe terminal")
for reference,observed in ((ra/name,oa/name),(rb/name,ob/name),(rg/name,og/name)):
    r=nib.load(str(reference));o=nib.load(str(observed))
    if (r.shape!=o.shape or not np.allclose(r.affine,o.affine,atol=1e-6)
        or not np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj))):
        raise SystemExit(f"three-route one-case mismatch: {observed}")
PY

wait_for_safe_host "three-route all-50 replay"
"${python_bin}" -u "${predictor}" --images "${images}" \
  --bbox-output "${full_bbox}" --cbg-output "${full_cbg}" \
  --guided-proposal "${proposal}" --guided-output "${full_guided}" \
  --terminal "${full_terminal}" --dataset-role public --gpu 0 \
  --part-id 0 --num-parts 1 --workers 0 --focus-batch-size 4 \
  >"${output}/full.log" 2>&1

"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${reference_guided}" "${full_bbox}" "${full_cbg}" "${full_guided}" \
  "${probe_terminal}" "${full_terminal}" "${predictor}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,rg,oa,ob,og,probe,full,predictor,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))
p=json.load(open(probe,encoding="utf-8"));f=json.load(open(full,encoding="utf-8"))
if (len(names)!=50 or f.get("status")!="COMPLETE" or f.get("case_count")!=50
    or not f.get("guided_route") or f.get("dataset_role")!="public"
    or f.get("public_labels_read") or f.get("hidden_read")
    or f.get("submission_created") or f.get("uploaded")):
    raise SystemExit("invalid three-route all-50 terminal")
rows=[];mismatches=[];geometry=[]
for name,row in zip(names,f["cases"]):
    current={"case":name,"wall_seconds":float(row["wall_seconds"])}
    for route,reference,observed in (
        ("bbox",ra/name,oa/name),("cbg",rb/name,ob/name),("guided",rg/name,og/name)):
        r=nib.load(str(reference));o=nib.load(str(observed))
        geom=r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6)
        equal=bool(np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)))
        current[f"{route}_geometry_equal"]=bool(geom);current[f"{route}_mask_equal"]=equal
        if not geom:geometry.append(f"{route}:{name}")
        if not equal:mismatches.append(f"{route}:{name}")
    current.update({key:int(row[key]) for key in (
      "grid_crops","bbox_crops","cbg_crops","guided_crops","proposal_crops","union_crops","reused_crops")})
    rows.append(current)
times=sorted(row["wall_seconds"] for row in rows)
def q(value):
 pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos))
 return times[lo]+(times[hi]-times[lo])*(pos-lo)
exact=not mismatches and not geometry
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_THREE_ROUTE_SHARED_FOCUS" if exact else "REJECT_THREE_ROUTE_SHARED_FOCUS",
 "deployment_authorized":False,"case_count":50,"exact_raw_route_equivalence":exact,
 "mismatches":mismatches,"geometry_failures":geometry,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "crop_totals":{key:sum(row[key] for row in rows) for key in (
   "grid_crops","bbox_crops","cbg_crops","guided_crops","proposal_crops","union_crops","reused_crops")},
 "predictor":{"path":str(predictor.resolve()),"sha256":sha(predictor)},
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "full_prediction":{"path":str(full.resolve()),"sha256":sha(full)},"cases":rows,
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps({k:payload[k] for k in ("decision","exact_raw_route_equivalence","runtime_seconds","crop_totals")},indent=2))
if not exact:raise SystemExit("three-route shared-Focus equivalence failed")
PY
