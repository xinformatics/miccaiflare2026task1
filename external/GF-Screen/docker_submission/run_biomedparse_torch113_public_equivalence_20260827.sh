#!/usr/bin/env bash
set -euo pipefail

# Validate the real BiomedParse prompt outputs under the challenge-compatible
# PyTorch 1.13.1/CUDA 11.6 image. This controller reads public images only,
# never labels or hidden data, and cannot package or upload a submission.
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
image="mayo-biomed-compat-probe:torch113"
expected_image_id="sha256:69b13dae0cfbf95dcc120aacd03d0c2c60039811fa4de66de48979175b13bbce"
upstream="${gf}/docker_submission/analysis/runtime_anchor_equivalence_20260827"
upstream_terminal="${upstream}/EQUIVALENCE.json"
upstream_lock="${upstream}/controller.lock"
inputs="${project}/external/BiomedParse/public_a160_240_inputs"
reference_whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_npz"
reference_tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_npz"
output="${gf}/docker_submission/analysis/biomedparse_torch113_public_equivalence_20260827"
probe="${output}/probe"
full="${output}/full"
probe_terminal="${probe}/PREDICT.json"
full_terminal="${full}/PREDICT.json"
terminal="${output}/EQUIVALENCE.json"

mkdir -p "${probe}/whole" "${probe}/tumor" "${full}/whole" "${full}/tumor"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "BiomedParse torch-1.13 equivalence already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

while [[ ! -s "${upstream_terminal}" ]]; do
  printf '%s waiting for complete raw-to-anchor runtime proof\n' "$(date -Ins)"
  sleep 30
done
exec 8>"${upstream_lock}"
flock 8
"${python_bin}" - "${upstream_terminal}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or not p.get("exact_anchor_equivalence")
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid runtime-anchor prerequisite")
PY

for directory in "${inputs}" "${reference_whole}" "${reference_tumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]] || {
    echo "BiomedParse inventory is not 50: ${directory}" >&2; exit 2;
  }
done
observed_image_id=$(docker image inspect "${image}" --format '{{.Id}}')
[[ "${observed_image_id}" == "${expected_image_id}" ]] || {
  echo "compatibility image identity changed" >&2; exit 2;
}

wait_for_safe_host() {
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
    printf '%s waiting for safe torch-1.13 probe host: available=%s pressure=%s gpu0=%s gpu1=%s\n' \
      "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used0}" "${used1}"
    sleep 30
  done
}

run_container() {
  local destination="$1"; shift
  docker run --rm --network none --gpus 'device=0' \
    -v "${project}:/project:ro" -v "${destination}:/output" \
    -w /project/external/BiomedParse/container_workspace \
    -e PYTHONPATH=/project/external/GF-Screen/docker_submission/vendor:/project/external/BiomedParse/container_workspace \
    -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    "${image}" python -u /project/external/BiomedParse/miccai_dual_prompt_inference.py \
    --input-dir /project/external/BiomedParse/public_a160_240_inputs \
    --whole-output-dir /output/whole --tumor-output-dir /output/tumor \
    --terminal /output/PREDICT.json \
    --checkpoint /project/external/GF-Screen/docker_submission/assets/biomedparse_3d_state_only.pt \
    --device cuda:0 --dataset-role public --slice-batch-size 4 \
    --existence-threshold 0.30 --existence-threshold 0.50 --existence-threshold 0.70 \
    "$@"
}

wait_for_safe_host
run_container "${probe}" --shard-id 14 --num-shards 50 >"${probe}/run.log" 2>&1
"${python_bin}" - "${reference_whole}" "${reference_tumor}" "${probe}" "${probe_terminal}" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
rw,rt,out,terminal=map(Path,sys.argv[1:]);p=json.load(open(terminal,encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("case_count")!=1 or p.get("model_loads")!=1
    or p.get("torch_version")!="1.13.1+cu116" or "/vendor/detectron2/" not in p.get("detectron2_module","")
    or p.get("peak_gpu_allocated_mb") is None or p.get("peak_gpu_reserved_mb") is None
    or p.get("public_labels_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid torch-1.13 BiomedParse probe terminal")
names=[path.name for path in (out/"whole").glob("*.npz")]
if len(names)!=1 or not (out/"tumor"/names[0]).is_file(): raise SystemExit("invalid probe inventory")
for reference,observed in ((rw/names[0],out/"whole"/names[0]),(rt/names[0],out/"tumor"/names[0])):
    with np.load(reference,allow_pickle=False) as a,np.load(observed,allow_pickle=False) as b:
        for key in ("segs","segs_e030","segs_e050","segs_e070","object_existence","existence_thresholds","prompts"):
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):
                raise SystemExit(f"torch-1.13 one-case mismatch: {observed}:{key}")
PY

wait_for_safe_host
run_container "${full}" >"${full}/run.log" 2>&1
"${python_bin}" - "${reference_whole}" "${reference_tumor}" "${full}" \
  "${probe_terminal}" "${full_terminal}" "${expected_image_id}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
rw,rt,out,probe,full,image_id,output=map(Path,sys.argv[1:])
image_id=str(image_id);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.load(open(probe,encoding="utf-8"));f=json.load(open(full,encoding="utf-8"))
names=sorted(path.name for path in rw.glob("*.npz"))
if (len(names)!=50 or f.get("status")!="COMPLETE" or f.get("case_count")!=50
    or f.get("model_loads")!=1 or f.get("torch_version")!="1.13.1+cu116"
    or "/vendor/detectron2/" not in f.get("detectron2_module","")
    or f.get("peak_gpu_allocated_mb") is None or f.get("peak_gpu_reserved_mb") is None
    or f.get("public_labels_read") or f.get("hidden_read")
    or f.get("submission_created") or f.get("uploaded")):
    raise SystemExit("invalid torch-1.13 BiomedParse all-50 terminal")
mismatches=[]
for name in names:
    for route,reference,observed in (("whole",rw/name,out/"whole"/name),("tumor",rt/name,out/"tumor"/name)):
        with np.load(reference,allow_pickle=False) as a,np.load(observed,allow_pickle=False) as b:
            for key in ("segs","segs_e030","segs_e050","segs_e070","object_existence","existence_thresholds","prompts"):
                if key not in a or key not in b or not np.array_equal(a[key],b[key]): mismatches.append(f"{route}:{name}:{key}")
times=sorted(float(row["wall_seconds"]) for row in f["cases"])
def q(value):
    pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos))
    return times[lo]+(times[hi]-times[lo])*(pos-lo)
exact=not mismatches
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_TORCH113_BIOMEDPARSE" if exact else "REJECT_TORCH113_BIOMEDPARSE",
 "deployment_authorized":False,"case_count":50,"exact_prompt_outputs":exact,
 "mismatches":mismatches,"torch_version":f["torch_version"],"docker_image_id":image_id,
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
if not exact: raise SystemExit("torch-1.13 BiomedParse all-50 equivalence failed")
PY
