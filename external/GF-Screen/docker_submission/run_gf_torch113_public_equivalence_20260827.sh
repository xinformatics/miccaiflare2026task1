#!/usr/bin/env bash
set -euo pipefail

# Validate the real compact bbox/CBG/guided GF outputs under the required
# Python-3.8/PyTorch-1.13 CUDA stack. This is public-image-only, serialized
# behind the BiomedParse compatibility replay, and cannot package or upload.
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
image="mayo-biomed-compat-probe:torch113"
expected_image_id="sha256:69b13dae0cfbf95dcc120aacd03d0c2c60039811fa4de66de48979175b13bbce"
upstream="${gf}/docker_submission/analysis/biomedparse_torch113_public_equivalence_20260827"
upstream_terminal="${upstream}/EQUIVALENCE.json"
upstream_lock="${upstream}/controller.lock"
images="${project}/validation/Validation-Public-Images"
proposal="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
reference_bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
reference_cbg="${gf}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/inference"
reference_guided="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
output="${gf}/docker_submission/analysis/gf_torch113_public_equivalence_20260827"
probe="${output}/probe"
full="${output}/full"
probe_terminal="${probe}/PREDICT.json"
full_terminal="${full}/PREDICT.json"
terminal="${output}/EQUIVALENCE.json"

mkdir -p "${probe}/bbox" "${probe}/cbg" "${probe}/guided" \
  "${full}/bbox" "${full}/cbg" "${full}/guided"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "GF torch-1.13 equivalence already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

while [[ ! -s "${upstream_terminal}" ]]; do
  printf '%s waiting for torch-1.13 BiomedParse equivalence\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${upstream_lock}"
flock 8
"${python_bin}" - "${upstream_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_prompt_outputs")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid torch-1.13 BiomedParse prerequisite")
PY

for directory in "${images}" "${proposal}" "${reference_bbox}" \
    "${reference_cbg}" "${reference_guided}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "GF fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
observed_image_id=$(docker image inspect "${image}" --format '{{.Id}}')
[[ "${observed_image_id}" == "${expected_image_id}" ]] || {
  echo "compatibility image identity changed" >&2; exit 2;
}

wait_for_safe_host() {
  local available_kib full_avg10 used0 used1
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
    printf '%s waiting for safe torch-1.13 GF host: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
      "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
    sleep 30
  done
}

run_container() {
  local destination="$1"; shift
  docker run --rm --network none --gpus 'device=0' \
    -v "${project}:/project:ro" -v "${destination}:/output" \
    -w /project/external/GF-Screen \
    -e PYTHONPATH=/project/external/GF-Screen/docker_submission/vendor:/project/external/GF-Screen \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    "${image}" python -u /project/external/GF-Screen/docker_submission/predict_dual_glance_shared_focus.py \
    --images /project/validation/Validation-Public-Images \
    --bbox-output /output/bbox --cbg-output /output/cbg \
    --guided-proposal /project/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti \
    --guided-output /output/guided --terminal /output/PREDICT.json \
    --bbox-checkpoint /project/external/GF-Screen/docker_submission/assets/gf_bbox_full_state.pt \
    --cbg-checkpoint /project/external/GF-Screen/docker_submission/assets/gf_cbg_glance_state.pt \
    --dataset-role public --gpu 0 --workers 0 --focus-batch-size 4 "$@"
}

wait_for_safe_host
run_container "${probe}" --part-id 14 --num-parts 50 >"${probe}/run.log" 2>&1
"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${reference_guided}" "${probe}" "${probe_terminal}" <<'PY'
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,rg,out,terminal=map(Path,sys.argv[1:])
name=sorted(p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz"))[14]
p=json.load(open(terminal,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1
    or p.get("python_version")!="3.8.0" or p.get("torch_version")!="1.13.1+cu116"
    or "/vendor/cc3d.py" not in p.get("cc3d_module","")
    or p.get("peak_gpu_allocated_mb") is None or p.get("peak_gpu_reserved_mb") is None
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid torch-1.13 GF probe terminal")
for reference,observed in ((ra/name,out/"bbox"/name),(rb/name,out/"cbg"/name),(rg/name,out/"guided"/name)):
    r=nib.load(str(reference));o=nib.load(str(observed))
    if (r.shape!=o.shape or not np.allclose(r.affine,o.affine,atol=1e-6)
        or not np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj))):
        raise SystemExit(f"torch-1.13 GF one-case mismatch: {observed}")
PY

wait_for_safe_host
run_container "${full}" --part-id 0 --num-parts 1 >"${full}/run.log" 2>&1
"${python_bin}" - "${images}" "${reference_bbox}" "${reference_cbg}" \
  "${reference_guided}" "${full}" "${probe_terminal}" "${full_terminal}" \
  "${expected_image_id}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import nibabel as nib
import numpy as np
images,ra,rb,rg,out,probe,full,image_id,output=map(Path,sys.argv[1:])
image_id=str(image_id);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.load(open(probe,encoding="utf-8"));f=json.load(open(full,encoding="utf-8"))
names=sorted(path.name.replace("_0000.nii.gz",".nii.gz") for path in images.glob("*_0000.nii.gz"))
if (len(names)!=50 or f.get("status")!="COMPLETE" or f.get("case_count")!=50
    or f.get("python_version")!="3.8.0" or f.get("torch_version")!="1.13.1+cu116"
    or "/vendor/cc3d.py" not in f.get("cc3d_module","")
    or f.get("peak_gpu_allocated_mb") is None or f.get("peak_gpu_reserved_mb") is None
    or f.get("public_labels_read") or f.get("hidden_read")
    or f.get("submission_created") or f.get("uploaded")):
    raise SystemExit("invalid torch-1.13 GF all-50 terminal")
mismatches=[];geometry=[]
for name in names:
    for route,reference,observed in (
        ("bbox",ra/name,out/"bbox"/name),("cbg",rb/name,out/"cbg"/name),("guided",rg/name,out/"guided"/name)):
        r=nib.load(str(reference));o=nib.load(str(observed))
        if r.shape!=o.shape or not np.allclose(r.affine,o.affine,atol=1e-6): geometry.append(f"{route}:{name}")
        if not np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)): mismatches.append(f"{route}:{name}")
times=sorted(float(row["wall_seconds"]) for row in f["cases"])
def q(value):
    pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos))
    return times[lo]+(times[hi]-times[lo])*(pos-lo)
exact=not mismatches and not geometry
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_TORCH113_GF" if exact else "REJECT_TORCH113_GF",
 "deployment_authorized":False,"case_count":50,"exact_raw_route_equivalence":exact,
 "mismatches":mismatches,"geometry_failures":geometry,
 "python_version":f["python_version"],"torch_version":f["torch_version"],
 "cc3d_module":f["cc3d_module"],"docker_image_id":image_id,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "peak_gpu_memory_mb":{"allocated":float(f["peak_gpu_allocated_mb"]),"reserved":float(f["peak_gpu_reserved_mb"])},
 "memory_eligible_16gb":float(f["peak_gpu_reserved_mb"])<=16384.0,
 "memory_eligible_4gb_efficiency":float(f["peak_gpu_reserved_mb"])<=4096.0,
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "full_prediction":{"path":str(full.resolve()),"sha256":sha(full)},
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps(payload,indent=2))
if not exact: raise SystemExit("torch-1.13 GF all-50 equivalence failed")
PY
