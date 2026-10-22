#!/usr/bin/env bash
set -euo pipefail

# Public-label-free sparse-STU exactness/runtime proof in the required
# Python-3.8/PyTorch-1.13 stack. It begins with the predeclared stress case and
# only spends the all-50 replay when that mask is exact and <=20 seconds.
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project}/external/STU-Net/pancancer"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
image="mayo-stunet-compat-probe:torch113"
expected_image_id="sha256:07e77e7203846b3e49afcbdd90600a583c745e28fb9fa8e241c3f20409bb8e23"
upstream="${gf}/docker_submission/analysis/gf_torch113_public_equivalence_20260827"
upstream_terminal="${upstream}/EQUIVALENCE.json"
upstream_lock="${upstream}/controller.lock"
dev_gate="${root}/public_sweeps/stunet_proxy_sparse_all50_20260827/EQUIVALENCE.json"
reference="${root}/public_sweeps/stunet_proxy_sparse_all50_20260827/reference"
images="${project}/validation/Validation-Public-Images"
anchor="${gf}/analysis/lesion_error_touchcomp_20260825/predictions"
bbox="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
model="${project}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="checkpoint_inference_epoch_0160.pth"
output="${gf}/docker_submission/analysis/stunet_torch113_sparse_equivalence_20260827"
probe="${output}/probe"
full="${output}/full"
probe_terminal="${probe}/PREDICT.json"
full_terminal="${full}/PREDICT.json"
terminal="${output}/EQUIVALENCE.json"
part_id=28

mkdir -p "${probe}/predictions" "${full}/predictions"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "STU torch-1.13 equivalence already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
fi

while [[ ! -s "${upstream_terminal}" || ! -s "${dev_gate}" ]]; do
  printf '%s waiting for torch-1.13 GF and development sparse-STU gates\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${upstream_lock}"
flock 8
"${python_bin}" - "${upstream_terminal}" "${dev_gate}" <<'PY'
import json,sys
gf,stu=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if (gf.get("status")!="COMPLETE" or not gf.get("exact_raw_route_equivalence")
    or stu.get("status")!="COMPLETE" or not stu.get("exact_full_volume_proxy_equivalence")
    or not stu.get("runtime_eligible_20s")
    or any(p.get("public_labels_read") or p.get("hidden_read")
           or p.get("submission_created") or p.get("uploaded") for p in (gf,stu))):
    raise SystemExit("invalid development runtime prerequisites")
PY
for directory in "${images}" "${anchor}" "${bbox}" "${whole}" "${tumor}" "${reference}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "STU torch-1.13 inventory is not 50: ${directory}" >&2; exit 2;
  }
done
observed_image_id=$(docker image inspect "${image}" --format '{{.Id}}')
[[ "${observed_image_id}" == "${expected_image_id}" ]] || {
  echo "STU compatibility image identity changed" >&2; exit 2;
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
    printf '%s waiting for safe torch-1.13 STU host: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
      "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
    sleep 30
  done
}

run_case() {
  local destination="$1" part="$2"
  docker run --rm --network none --gpus 'device=0' \
    -v "${project}:/project:ro" -v "${destination}:/output" \
    -w /project/external/STU-Net/pancancer \
    -e PYTHONPATH=/project/external/STU-Net/pancancer:/project/external/GF-Screen/docker_submission/vendor:/project/external/GF-Screen/docker_submission:/project/external/GF-Screen \
    -e nnUNet_compile=false -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    "${image}" python -u predict_stunet_proxy_sparse.py \
    --images /project/validation/Validation-Public-Images \
    --anchors /project/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions \
    --bbox-raw /project/external/GF-Screen/predictions/gf_a055_bbox_pu_best_ep280/public_raw \
    --prompt-whole /project/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti \
    --prompt-tumor /project/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti \
    --output /output/predictions \
    --model-dir /project/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox \
    --checkpoint "${checkpoint}" --gpu 0 --part-id "${part}" --num-parts 50 \
    --workers 1 --dataset-role public --terminal /output/PREDICT.json
}

compare_case() {
  "${python_bin}" - "$1" "$2" "$3" <<'PY'
import json,sys
from pathlib import Path
import nibabel as nib
import numpy as np
reference,observed,terminal=map(Path,sys.argv[1:]);p=json.load(open(terminal,encoding="utf-8"));rows=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1 or len(rows)!=1
    or p.get("peak_gpu_allocated_mb") is None or p.get("peak_gpu_reserved_mb") is None
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid torch-1.13 STU case terminal")
name=rows[0]["case"];r=nib.load(str(reference/name));o=nib.load(str(observed/name))
exact=(r.shape==o.shape and np.allclose(r.affine,o.affine,atol=1e-6)
       and np.array_equal(np.asanyarray(r.dataobj),np.asanyarray(o.dataobj)))
raise SystemExit(0 if exact and float(p["wall_seconds"])<=20.0 else 1)
PY
}

wait_for_safe_host
run_case "${probe}" "${part_id}" >"${probe}/run.log" 2>&1
if ! compare_case "${reference}" "${probe}/predictions" "${probe_terminal}"; then
  "${python_bin}" - "${probe_terminal}" "${expected_image_id}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
source,image_id,output=map(Path,sys.argv[1:]);p=json.load(open(source,encoding="utf-8"))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"REJECT_TORCH113_STUNET_AT_PROBE","deployment_authorized":False,
 "probe_wall_seconds":p.get("wall_seconds"),"docker_image_id":str(image_id),
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp");temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
PY
  exit 0
fi

for part in $(seq 0 49); do
  case4=$(printf '%04d' "${part}")
  destination="${full}/case_${case4}"
  mkdir -p "${destination}/predictions"
  wait_for_safe_host
  run_case "${destination}" "${part}" >"${destination}/run.log" 2>&1
  compare_case "${reference}" "${destination}/predictions" "${destination}/PREDICT.json"
done

"${python_bin}" - "${full}" "${expected_image_id}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
root,image_id,output=map(Path,sys.argv[1:]);times=[];rows=[]
for index in range(50):
    terminal=root/f"case_{index:04d}"/"PREDICT.json";p=json.load(open(terminal,encoding="utf-8"));row=p["cases"][0]
    times.append(float(p["wall_seconds"]));rows.append({"case":row["case"],"wall_seconds":float(p["wall_seconds"]),
      "stage_seconds":float(row["seconds"]),"stunet_skipped":bool(row["stunet_skipped"]),
      "full_tiles":int(row["full_tiles"]),"selected_tiles":int(row["selected_tiles"]),
      "model_initialization_seconds":float(p["model_initialization_seconds"]),
      "peak_gpu_allocated_mb":float(p["peak_gpu_allocated_mb"]),
      "peak_gpu_reserved_mb":float(p["peak_gpu_reserved_mb"])})
times.sort()
def q(value):
    pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos));return times[lo]+(times[hi]-times[lo])*(pos-lo)
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_TORCH113_STUNET","deployment_authorized":False,
 "case_count":50,"exact_full_volume_proxy_equivalence":True,"runtime_eligible_20s":max(times)<=20,
 "runtime_seconds":{"mean":sum(times)/50,"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "peak_gpu_memory_mb":{"allocated":max(row["peak_gpu_allocated_mb"] for row in rows),
   "reserved":max(row["peak_gpu_reserved_mb"] for row in rows)},
 "model_initialization_seconds":max(row["model_initialization_seconds"] for row in rows),
 "memory_eligible_16gb":max(row["peak_gpu_reserved_mb"] for row in rows)<=16384.0,
 "memory_eligible_4gb_efficiency":max(row["peak_gpu_reserved_mb"] for row in rows)<=4096.0,
 "stunet_skipped_cases":sum(row["stunet_skipped"] for row in rows),"cases":rows,
 "docker_image_id":str(image_id),"python_version":"3.8.0","torch_version":"1.13.1+cu116","nnunetv2_version":"2.4.2",
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp");temporary.write_text(json.dumps(payload,indent=2)+"\n");os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
