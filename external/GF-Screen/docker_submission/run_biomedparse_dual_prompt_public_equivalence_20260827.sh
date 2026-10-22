#!/usr/bin/env bash
set -euo pipefail

# Public-label-free equivalence proof for one persistent BiomedParse model with
# the winner's two separate prompts. It queues behind the shared-Focus replay,
# probes one deterministic case, then runs all 50 only after exact agreement.

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
gf="${project}/external/GF-Screen"
biomed="${project}/external/BiomedParse"
runner="${biomed}/miccai_dual_prompt_inference.py"
official_env="${biomed}/run_official_env.sh"
audit_python="/home/m348911/miniconda3/envs/radoncAI/bin/python"
inputs="${biomed}/public_a160_240_inputs"
reference_whole="${biomed}/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_npz"
reference_tumor="${biomed}/public_sweeps/automatic_text_proposals_20260820/tumor_npz"
shared_lock="${gf}/docker_submission/analysis/dual_glance_shared_focus_public_equivalence_20260827/controller.lock"
output="${gf}/docker_submission/analysis/biomedparse_dual_prompt_public_equivalence_20260827"
probe_whole="${output}/probe/whole"
probe_tumor="${output}/probe/tumor"
full_whole="${output}/full/whole"
full_tumor="${output}/full/tumor"
probe_terminal="${output}/PROBE_PREDICT.json"
full_terminal="${output}/FULL_PREDICT.json"
terminal="${output}/EQUIVALENCE.json"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
mkdir -p "${output}" "${probe_whole}" "${probe_tumor}" "${full_whole}" "${full_tumor}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "BiomedParse dual-prompt equivalence already active" >&2; exit 1; }

# The shared-Focus controller holds its lock for its complete queued lifetime.
exec 8>"${shared_lock}"
flock 8

for directory in "${inputs}" "${reference_whole}" "${reference_tumor}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.npz' | wc -l) -eq 50 ]] || {
    echo "fixed-public inventory is not 50: ${directory}" >&2; exit 2;
  }
done
[[ -x "${official_env}" && -s "${runner}" ]]
while :; do
  used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] && break
  printf '%s waiting for idle GPU 0: %s MiB\n' "$(date -Ins)" "${used}"
  sleep 30
done

"${official_env}" -u "${runner}" --input-dir "${inputs}" \
  --whole-output-dir "${probe_whole}" --tumor-output-dir "${probe_tumor}" \
  --terminal "${probe_terminal}" --dataset-role public --device cuda:0 \
  --shard-id 14 --num-shards 50 --slice-batch-size 4 \
  --existence-threshold 0.30 --existence-threshold 0.50 --existence-threshold 0.70 \
  >"${output}/probe.log" 2>&1

"${audit_python}" - "${reference_whole}" "${reference_tumor}" \
  "${probe_whole}" "${probe_tumor}" "${probe_terminal}" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
rw,rt,ow,ot,terminal=map(Path,sys.argv[1:])
payload=json.load(open(terminal,encoding="utf-8"))
if (payload.get("status")!="COMPLETE" or payload.get("case_count")!=1
    or payload.get("model_loads")!=1
    or payload.get("prompt_execution")!="two_separate_forward_passes"
    or payload.get("public_labels_read") or payload.get("hidden_read")
    or payload.get("submission_created") or payload.get("uploaded")):
    raise SystemExit("invalid BiomedParse dual-prompt probe terminal")
names=[p.name for p in ow.glob("*.npz")]
if len(names)!=1 or not (ot/names[0]).is_file():
    raise SystemExit("invalid BiomedParse dual-prompt probe inventory")
for reference,observed in ((rw/names[0],ow/names[0]),(rt/names[0],ot/names[0])):
    with np.load(reference,allow_pickle=False) as a,np.load(observed,allow_pickle=False) as b:
        for key in ("segs","segs_e030","segs_e050","segs_e070","object_existence","existence_thresholds","prompts"):
            if key not in a or key not in b or not np.array_equal(a[key],b[key]):
                raise SystemExit(f"BiomedParse one-case mismatch: {observed}:{key}")
PY

"${official_env}" -u "${runner}" --input-dir "${inputs}" \
  --whole-output-dir "${full_whole}" --tumor-output-dir "${full_tumor}" \
  --terminal "${full_terminal}" --dataset-role public --device cuda:0 \
  --slice-batch-size 4 \
  --existence-threshold 0.30 --existence-threshold 0.50 --existence-threshold 0.70 \
  >"${output}/full.log" 2>&1

"${audit_python}" - "${reference_whole}" "${reference_tumor}" \
  "${full_whole}" "${full_tumor}" "${probe_terminal}" "${full_terminal}" "${terminal}" <<'PY'
import datetime,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
rw,rt,ow,ot,probe,full,output=map(Path,sys.argv[1:])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.load(open(probe,encoding="utf-8"));f=json.load(open(full,encoding="utf-8"))
names=sorted(path.name for path in rw.glob("*.npz"))
if (len(names)!=50 or f.get("status")!="COMPLETE" or f.get("case_count")!=50
    or f.get("model_loads")!=1 or f.get("dataset_role")!="public"
    or f.get("public_labels_read") or f.get("hidden_read")
    or f.get("submission_created") or f.get("uploaded")):
    raise SystemExit("invalid BiomedParse dual-prompt full terminal")
mismatches=[];rows=[]
for name,row in zip(names,f["cases"]):
    current={"case":name,"wall_seconds":float(row["wall_seconds"])}
    for route,reference,observed in (("whole",rw/name,ow/name),("tumor",rt/name,ot/name)):
        with np.load(reference,allow_pickle=False) as a,np.load(observed,allow_pickle=False) as b:
            equal=True
            for key in ("segs","segs_e030","segs_e050","segs_e070","object_existence","existence_thresholds","prompts"):
                equal &= key in a and key in b and np.array_equal(a[key],b[key])
        current[f"{route}_exact"]=bool(equal)
        if not equal:mismatches.append(f"{route}:{name}")
    rows.append(current)
times=sorted(row["wall_seconds"] for row in rows)
def q(value):
 pos=(len(times)-1)*value;lo=int(math.floor(pos));hi=int(math.ceil(pos))
 return times[lo]+(times[hi]-times[lo])*(pos-lo)
exact=not mismatches
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE" if exact else "FAILED",
 "decision":"QUALIFY_PERSISTENT_DUAL_PROMPT" if exact else "REJECT_PERSISTENT_DUAL_PROMPT",
 "deployment_authorized":False,"case_count":50,"exact_prompt_outputs":exact,
 "mismatches":mismatches,
 "runtime_seconds":{"mean":sum(times)/len(times),"median":q(.5),"p90":q(.9),"p95":q(.95),"max":max(times)},
 "model_loads":1,"prompt_execution":"two_separate_forward_passes",
 "probe":{"path":str(probe.resolve()),"sha256":sha(probe)},
 "full_prediction":{"path":str(full.resolve()),"sha256":sha(full)},"cases":rows,
 "public_images_read":True,"public_labels_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
print(json.dumps({k:payload[k] for k in ("decision","exact_prompt_outputs","runtime_seconds")},indent=2))
if not exact:raise SystemExit("BiomedParse dual-prompt full equivalence failed")
PY
