#!/usr/bin/env bash
set -euo pipefail

# Full fixed-public equivalence/runtime gate for sparse STU p70 confirmation.
# Requires the frozen one-case probe first. It has no hidden, label, packaging,
# or upload path and never changes the trained checkpoint.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
freetumor="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
baseline_root="${root}/public_sweeps/stunetb_matched_e160_20260826"
baseline_stu="${baseline_root}/component_predictions"
baseline_terminal="${baseline_root}/TERMINAL.json"
probe_terminal="${root}/public_sweeps/stunet_anchor_sparse_e160_probe_20260827/EQUIVALENCE.json"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
output="${root}/public_sweeps/stunet_anchor_sparse_e160_all50_20260827"
predictions="${output}/predictions"
terminal="${output}/EQUIVALENCE.json"
gpu="${SPARSE_EQUIVALENCE_GPU:-0}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-sparse
[[ "${gpu}" =~ ^[01]$ ]] || { echo "SPARSE_EQUIVALENCE_GPU must be 0 or 1" >&2; exit 2; }
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "sparse all-50 equivalence already active" >&2; exit 1; }

for directory in "${images}" "${anchor}" "${freetumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
[[ -s "${checkpoint}" && -s "${baseline_terminal}" && -s "${probe_terminal}" ]]
"${python_bin}" - "${probe_terminal}" "${baseline_terminal}" "${checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
probe,baseline,checkpoint=map(Path,sys.argv[1:]);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.load(open(probe,encoding="utf-8"));b=json.load(open(baseline,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("probe_pass") or not p.get("mask_equal")
    or not p.get("runtime_eligible_55s") or p.get("deployment_authorized")
    or p.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
    or p.get("public_labels_read") or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("one-case sparse probe is not qualified")
if (b.get("status")!="COMPLETE" or b.get("evaluation_epoch")!=160
    or b.get("component_checkpoint",{}).get("sha256")!=sha(checkpoint)
    or b.get("hidden_read") or b.get("submission_created") or b.get("uploaded")):
    raise SystemExit("invalid immutable epoch-160 baseline")
PY

used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] || {
  echo "GPU ${gpu} is occupied (${used} MiB); all-50 equivalence not started" >&2
  exit 3
}

checkpoint_sha=$(sha256sum "${checkpoint}" | awk '{print $1}')
for part_id in $(seq 0 49); do
  part4=$(printf '%04d' "${part_id}")
  case_terminal="${output}/CASE_${part4}.json"
  if [[ -s "${case_terminal}" ]] && "${python_bin}" - "${case_terminal}" "${checkpoint_sha}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"));row=(p.get("cases") or [{}])[0]
valid=(p.get("status")=="COMPLETE" and p.get("case_count")==1
       and p.get("checkpoint",{}).get("sha256")==sys.argv[2]
       and p.get("dataset_role")=="public" and not p.get("public_labels_read")
       and not p.get("hidden_read") and not p.get("submission_created") and not p.get("uploaded")
       and isinstance(p.get("wall_seconds"),float) and row.get("case"))
raise SystemExit(0 if valid else 1)
PY
  then
    continue
  fi
  timeout --signal=TERM 180s "${python_bin}" -u "${root}/predict_stunet_anchor_sparse.py" \
    --images "${images}" --anchors "${anchor}" --freetumor "${freetumor}" \
    --output "${predictions}" --model-dir "${model}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu "${gpu}" \
    --part-id "${part_id}" --num-parts 50 --workers 1 \
    --dataset-role public --terminal "${case_terminal}" \
    >"${output}/case_${part4}.log" 2>&1
done

[[ $(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ "${checkpoint_sha}" == "$(sha256sum "${checkpoint}" | awk '{print $1}')" ]]

"${python_bin}" - "${images}" "${anchor}" "${freetumor}" "${baseline_stu}" \
  "${predictions}" "${output}" "${checkpoint}" "${probe_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
sys.path.insert(0,str(Path(sys.argv[2]).parents[3]/"STU-Net"/"pancancer"))
from analyze_stunet_gf_ensemble import confirmed_anchor_components
images,anchor,free,baseline,predictions,root,checkpoint,probe,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=sorted(path.name.replace("_0000.nii.gz",".nii.gz") for path in images.glob("*_0000.nii.gz"))
if len(names)!=50: raise SystemExit("public image inventory drift")
rows=[];mismatches=[];geometry_failures=[]
for index,name in enumerate(names):
    case_terminal=root/f"CASE_{index:04d}.json";p=json.load(open(case_terminal,encoding="utf-8"));case_rows=p.get("cases") or []
    if (p.get("status")!="COMPLETE" or p.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
        or p.get("case_count")!=1 or len(case_rows)!=1 or case_rows[0].get("case")!=name
        or p.get("public_labels_read") or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
        raise SystemExit(f"invalid sparse case terminal {index}")
    aimg=nib.load(str(anchor/name));fimg=nib.load(str(free/name));simg=nib.load(str(predictions/name))
    a=np.asanyarray(aimg.dataobj).astype(bool);f=np.asanyarray(fimg.dataobj).astype(bool)
    with np.load(baseline/f"{name.removesuffix('.nii.gz')}.prob16.npz") as archive:
        probability=archive["lesion_probability_uint16"].astype(np.float32)/65535.0
    expected=confirmed_anchor_components(a,probability>0.70,f,"either")
    observed=np.asanyarray(simg.dataobj).astype(bool)
    equal=bool(np.array_equal(expected,observed));geometry=(simg.shape==aimg.shape and np.allclose(simg.affine,aimg.affine,atol=1e-6))
    if not equal: mismatches.append(name)
    if not geometry: geometry_failures.append(name)
    row=case_rows[0]
    rows.append({"case":name,"mask_equal":equal,"geometry_equal":bool(geometry),
      "wall_seconds":float(p["wall_seconds"]),"model_initialization_seconds":float(p["model_initialization_seconds"]),
      "stunet_skipped":bool(row["stunet_skipped"]),"full_tiles":int(row["full_tiles"]),
      "selected_tiles":int(row["selected_tiles"])})
times=sorted(row["wall_seconds"] for row in rows)
def percentile(q):
    position=(len(times)-1)*q;lower=int(math.floor(position));upper=int(math.ceil(position))
    return times[lower]+(times[upper]-times[lower])*(position-lower)
exact=not mismatches and not geometry_failures
runtime=max(times)<=55.0
authorized=bool(exact and runtime)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"AUTHORIZE_SPARSE_DEPLOYMENT" if authorized else "REJECT_SPARSE_DEPLOYMENT",
 "deployment_authorized":authorized,"case_count":50,"exact_final_mask_equivalence":exact,
 "mismatched_cases":mismatches,"geometry_failures":geometry_failures,"runtime_eligible_55s":runtime,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":percentile(.5),"p90":percentile(.9),
                    "p95":percentile(.95),"max":max(times)},
 "stunet_skipped_cases":sum(row["stunet_skipped"] for row in rows),
 "full_tiles":sum(row["full_tiles"] for row in rows),"selected_tiles":sum(row["selected_tiles"] for row in rows),
 "cases":rows,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps({key:payload[key] for key in ("decision","exact_final_mask_equivalence","runtime_eligible_55s","runtime_seconds","stunet_skipped_cases","full_tiles","selected_tiles")},indent=2))
PY
