#!/usr/bin/env bash
set -euo pipefail

# Public-label-free equivalence proof for exact dual-Glance/shared-Focus replay.
# It queues behind the sparse-STU runtime controller on GPU 0, probes one case,
# and runs all 50 only after both route masks match their immutable raw caches.

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
stunet="${project}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${gf}/docker_submission/predict_dual_glance_shared_focus.py"
images="${project}/validation/Validation-Public-Images"
reference_bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
reference_cbg="${gf}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/inference"
sparse_lock="${stunet}/runs/stunet_anchor_sparse_after_sourcestrat_20260827/controller.lock"
output="${gf}/docker_submission/analysis/dual_glance_shared_focus_public_equivalence_20260827"
probe_bbox="${output}/probe/bbox"
probe_cbg="${output}/probe/cbg"
full_bbox="${output}/full/bbox"
full_cbg="${output}/full/cbg"
probe_terminal="${output}/PROBE_PREDICT.json"
full_terminal="${output}/FULL_PREDICT.json"
terminal="${output}/EQUIVALENCE.json"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-shared-focus MALLOC_ARENA_MAX=2
mkdir -p "${output}" "${probe_bbox}" "${probe_cbg}" "${full_bbox}" "${full_cbg}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "shared-Focus equivalence already active" >&2; exit 1; }

# Block behind the already-armed sparse runtime proof without polling or racing
# it for GPU 0. The sparse controller holds this lock for its complete lifetime.
exec 8>"${sparse_lock}"
flock 8

for directory in "${images}" "${reference_bbox}" "${reference_cbg}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
[[ -s "${predictor}" ]]
while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && break
  printf '%s waiting for idle GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done

"${python_bin}" -u "${predictor}" --images "${images}" \
  --bbox-output "${probe_bbox}" --cbg-output "${probe_cbg}" \
  --terminal "${probe_terminal}" --dataset-role public --gpu 0 \
  --part-id 14 --num-parts 50 --workers 0 --focus-batch-size 4 \
  >"${output}/probe.log" 2>&1

"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${probe_bbox}" "${probe_cbg}" "${probe_terminal}" <<'PY'
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,oa,ob,terminal=map(Path,sys.argv[1:])
names=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))
name=names[14]; p=json.load(open(terminal,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1
    or p.get("dataset_role")!="public" or p.get("public_labels_read")
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid shared-Focus probe terminal")
for reference,observed in ((ra/name,oa/name),(rb/name,ob/name)):
    r=nib.load(str(reference));o=nib.load(str(observed))
    if (r.shape!=o.shape or not np.allclose(r.affine,o.affine,atol=1e-6)
        or not np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj))):
        raise SystemExit(f"shared-Focus one-case mismatch: {observed}")
PY

"${python_bin}" -u "${predictor}" --images "${images}" \
  --bbox-output "${full_bbox}" --cbg-output "${full_cbg}" \
  --terminal "${full_terminal}" --dataset-role public --gpu 0 \
  --part-id 0 --num-parts 1 --workers 0 --focus-batch-size 4 \
  >"${output}/full.log" 2>&1

"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${full_bbox}" "${full_cbg}" "${probe_terminal}" "${full_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,oa,ob,probe,full,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))
p=json.load(open(probe,encoding="utf-8"));f=json.load(open(full,encoding="utf-8"))
if (len(names)!=50 or f.get("status")!="COMPLETE" or f.get("case_count")!=50
    or f.get("dataset_role")!="public" or f.get("public_labels_read")
    or f.get("hidden_read") or f.get("submission_created") or f.get("uploaded")):
    raise SystemExit("invalid shared-Focus full terminal")
rows=[];mismatches=[];geometry=[]
for name,row in zip(names,f["cases"]):
    current={"case":name,"wall_seconds":float(row["wall_seconds"])}
    for route,reference,observed in (("bbox",ra/name,oa/name),("cbg",rb/name,ob/name)):
        r=nib.load(str(reference));o=nib.load(str(observed))
        geom=r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6)
        equal=bool(np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)))
        current[f"{route}_geometry_equal"]=bool(geom);current[f"{route}_mask_equal"]=equal
        if not geom: geometry.append(f"{route}:{name}")
        if not equal: mismatches.append(f"{route}:{name}")
    current.update({key:int(row[key]) for key in ("grid_crops","bbox_crops","cbg_crops","union_crops","reused_crops")})
    rows.append(current)
times=sorted(row["wall_seconds"] for row in rows)
def q(value):
 pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos))
 return times[lo]+(times[hi]-times[lo])*(pos-lo)
exact=not mismatches and not geometry
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_SHARED_FOCUS_FOR_WINNER_INTEGRATION" if exact else "REJECT_SHARED_FOCUS",
 "deployment_authorized":False,"case_count":50,"exact_raw_route_equivalence":exact,
 "mismatches":mismatches,"geometry_failures":geometry,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "crop_totals":{key:sum(row[key] for row in rows) for key in ("grid_crops","bbox_crops","cbg_crops","union_crops","reused_crops")},
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "full_prediction":{"path":str(full.resolve()),"sha256":sha(full)},"cases":rows,
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps({k:payload[k] for k in ("decision","exact_raw_route_equivalence","runtime_seconds","crop_totals")},indent=2))
if not exact: raise SystemExit("shared-Focus full equivalence failed")
PY

