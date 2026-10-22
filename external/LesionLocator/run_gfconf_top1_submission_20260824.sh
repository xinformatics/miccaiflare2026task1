#!/usr/bin/env bash
set -euo pipefail

# Deadline-safe full-public gate followed by label-free hidden inference.
# This controller never uploads and never changes DEPLOYMENT_CHAMPION.json.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
champion="${workspace}/external/GF-Screen/analysis/DEPLOYMENT_CHAMPION.json"
smoke_terminal="${script_dir}/public_audit/gfconf_top1_fold0_notta_smoke_20260824/SMOKE_TERMINAL.json"
public_images="${workspace}/validation/Validation-Public-Images"
public_labels="${workspace}/validation/Validation-Public-Labels"
hidden_images="${workspace}/validation/Validation-Hidden-Images"
model_root="${script_dir}/LesionLocatorCheckpoint"
output="${script_dir}/full_runs/gfconf_top1_fold0_notta_20260824"
terminal="${output}/TERMINAL.json"
archive="${workspace}/submissions/gf_ll_top1_cbg41.zip"
lock="${output}/controller.lock"

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "gf_conf top-1 full controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

for required in "${champion}" "${public_images}" "${public_labels}" \
  "${hidden_images}" "${model_root}/LesionLocatorSeg/bbox_optimized/fold_0/checkpoint_final.pth"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

mapfile -t anchor_paths < <("${python_bin}" - "${champion}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"))
public=Path(p["public_predictions"]); hidden=Path(p["hidden_predictions"])
metrics=Path(p["public_metrics"]); archive=Path(p["archive"])
if not public.is_dir() or not hidden.is_dir() or not metrics.is_file() or not archive.is_file():
 raise SystemExit("champion assets missing")
if hashlib.sha256(metrics.read_bytes()).hexdigest()!=p["public_metrics_sha256"]:
 raise SystemExit("champion public-metric identity mismatch")
if hashlib.sha256(archive.read_bytes()).hexdigest()!=p["archive_sha256"]:
 raise SystemExit("champion archive identity mismatch")
print(public.resolve()); print(hidden.resolve())
PY
)
[[ "${#anchor_paths[@]}" -eq 2 ]] || exit 2
public_anchor="${anchor_paths[0]}"
hidden_anchor="${anchor_paths[1]}"

mkdir -p "${output}/public_prompts" "${output}/hidden_prompts" \
  "${output}/public_instances" "${output}/hidden_instances" \
  "${output}/public_predictions" "${output}/hidden_predictions" \
  "${output}/metrics"

# Prompt construction is label-free and runs while the five-case smoke is on GPU.
"${python_bin}" -u "${script_dir}/prepare_topk_component_prompts.py" \
  --input "${public_anchor}" --output "${output}/public_prompts" \
  --audit "${output}/public_prompt_audit.json" --top-k 1 --min-mm3 0 \
  --max-mm3 inf --expected-count 50 > "${output}/public_prepare.log" 2>&1 &
public_prepare_pid=$!
"${python_bin}" -u "${script_dir}/prepare_topk_component_prompts.py" \
  --input "${hidden_anchor}" --output "${output}/hidden_prompts" \
  --audit "${output}/hidden_prompt_audit.json" --top-k 1 --min-mm3 0 \
  --max-mm3 inf --expected-count 100 > "${output}/hidden_prepare.log" 2>&1 &
hidden_prepare_pid=$!
wait "${public_prepare_pid}"
wait "${hidden_prepare_pid}"

while [[ ! -s "${smoke_terminal}" ]]; do
  printf '%s waiting for five-case smoke terminal\n' "$(date -Ins)"
  sleep 15
done
smoke_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${smoke_terminal}")
if [[ "${smoke_decision}" != "QUALIFY_FOR_FULL_PUBLIC" ]]; then
  "${python_bin}" - "${smoke_terminal}" "${champion}" "${terminal}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
smoke,champion,out=map(Path,sys.argv[1:])
p={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
   "status":"COMPLETE","decision":"PRUNE_AFTER_SMOKE",
   "smoke_terminal":str(smoke.resolve()),"smoke_terminal_sha256":hashlib.sha256(smoke.read_bytes()).hexdigest(),
   "champion_manifest":str(champion.resolve()),"submission_created":False,"uploaded":False}
tmp=out.with_name(f".{out.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(p,indent=2)+"\n");os.replace(tmp,out)
PY
  exit 0
fi

select_gpu() {
  while :; do
    mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [[ "${#used[@]}" -eq 2 && "${used[0]}" =~ ^[0-9]+$ && "${used[1]}" =~ ^[0-9]+$ ]] || exit 2
    if [[ "${used[0]}" -le 16384 || "${used[1]}" -le 16384 ]]; then
      if [[ "${used[0]}" -le "${used[1]}" ]]; then echo 0; else echo 1; fi
      return
    fi
    printf '%s waiting for GPU below 16 GiB: gpu0=%s gpu1=%s\n' \
      "$(date -Ins)" "${used[0]}" "${used[1]}" >&2
    sleep 30
  done
}

export MKL_THREADING_LAYER=GNU
gpu=$(select_gpu)
public_start=$(date +%s)
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${script_dir}" "${python_bin}" -u \
  -c 'from lesionlocator.inference.lesionlocator_segment import predict_seg_from_prompt; predict_seg_from_prompt()' \
  -i "${public_images}" -p "${output}/public_prompts" -t box \
  -o "${output}/public_instances" -m "${model_root}" -f 0 \
  --disable_tta --disable_progress_bar -npp 2 -nps 2 \
  > "${output}/public_inference.log" 2>&1
public_seconds=$(($(date +%s)-public_start))

"${python_bin}" -u "${script_dir}/materialize_topk_prompt_replacement.py" \
  --references "${public_images}" --instances "${output}/public_instances" \
  --anchor "${public_anchor}" --prompts "${output}/public_prompts" \
  --output "${output}/public_predictions" --audit "${output}/public_replacement_audit.json" \
  --expected-count 50 --min-seed-coverage 0.25 --min-refined-overlap 0.25 \
  --min-volume-ratio 0.25 --max-volume-ratio 4.0 \
  > "${output}/public_materialize.log" 2>&1
"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${public_anchor}" --gt "${public_labels}" \
  --csv "${output}/metrics/anchor.csv" --workers 4 > "${output}/metrics/anchor.log" 2>&1
"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${output}/public_predictions" --gt "${public_labels}" \
  --csv "${output}/metrics/candidate.csv" --workers 4 > "${output}/metrics/candidate.log" 2>&1

public_gate=$("${python_bin}" - "${output}/metrics/anchor.csv" "${output}/metrics/candidate.csv" <<'PY'
import csv,sys
def read(path):
 with open(path,newline="",encoding="utf-8") as f:r=list(csv.DictReader(f))
 if len(r)!=50:raise SystemExit("exact 50-case public metric coverage required")
 return r
a,c=read(sys.argv[1]),read(sys.argv[2])
mean=lambda rows,key:sum(float(x[key]) for x in rows)/len(rows)
ad,an=mean(a,"Lesion_DSC"),mean(a,"Lesion_NSD")
cd,cn=mean(c,"Lesion_DSC"),mean(c,"Lesion_NSD")
print("PROMOTE" if cd>=ad+0.0005 and cn>=an-0.002 else "PRUNE")
print(ad,an,cd,cn,file=sys.stderr)
PY
)
if [[ "${public_gate}" != "PROMOTE" ]]; then
  "${python_bin}" - "${output}/metrics/anchor.csv" "${output}/metrics/candidate.csv" \
    "${smoke_terminal}" "${champion}" "${public_seconds}" "${terminal}" <<'PY'
import csv,hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
a,c,s,h=map(Path,sys.argv[1:5]);seconds=int(sys.argv[5]);out=Path(sys.argv[6])
def means(p):
 with p.open(newline="") as f:r=list(csv.DictReader(f))
 return {k:sum(float(x[k]) for x in r)/len(r) for k in ("Lesion_DSC","Lesion_NSD")}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),"status":"COMPLETE",
   "decision":"PRUNE_AFTER_FULL_PUBLIC","anchor":means(a),"candidate":means(c),
   "public_inference_seconds":seconds,"smoke_terminal":str(s.resolve()),"smoke_terminal_sha256":sha(s),
   "champion_manifest":str(h.resolve()),"submission_created":False,"uploaded":False}
tmp=out.with_name(f".{out.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(p,indent=2)+"\n");os.replace(tmp,out)
PY
  exit 0
fi

gpu=$(select_gpu)
hidden_start=$(date +%s)
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${script_dir}" "${python_bin}" -u \
  -c 'from lesionlocator.inference.lesionlocator_segment import predict_seg_from_prompt; predict_seg_from_prompt()' \
  -i "${hidden_images}" -p "${output}/hidden_prompts" -t box \
  -o "${output}/hidden_instances" -m "${model_root}" -f 0 \
  --disable_tta --disable_progress_bar -npp 2 -nps 2 \
  > "${output}/hidden_inference.log" 2>&1
hidden_seconds=$(($(date +%s)-hidden_start))

"${python_bin}" -u "${script_dir}/materialize_topk_prompt_replacement.py" \
  --references "${hidden_images}" --instances "${output}/hidden_instances" \
  --anchor "${hidden_anchor}" --prompts "${output}/hidden_prompts" \
  --output "${output}/hidden_predictions" --audit "${output}/hidden_replacement_audit.json" \
  --expected-count 100 --min-seed-coverage 0.25 --min-refined-overlap 0.25 \
  --min-volume-ratio 0.25 --max-volume-ratio 4.0 \
  > "${output}/hidden_materialize.log" 2>&1

"${python_bin}" -u "${workspace}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/hidden_predictions" \
  --output "${archive}" --expected-count 100 > "${output}/package.log" 2>&1

"${python_bin}" - "${output}/metrics/anchor.csv" "${output}/metrics/candidate.csv" \
  "${output}/public_replacement_audit.json" "${output}/hidden_replacement_audit.json" \
  "${smoke_terminal}" "${champion}" "${archive}" "${public_seconds}" "${hidden_seconds}" "${terminal}" <<'PY'
import csv,hashlib,json,os,sys,zipfile
from datetime import datetime
from pathlib import Path
a,c,pa,ha,s,h,z=map(Path,sys.argv[1:8]);ps,hs=map(int,sys.argv[8:10]);out=Path(sys.argv[10])
def means(p):
 with p.open(newline="") as f:r=list(csv.DictReader(f))
 return {k:sum(float(x[k]) for x in r)/len(r) for k in ("Lesion_DSC","Lesion_NSD")}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(z) as q:
 names=q.namelist();bad=q.testzip()
if len(names)!=100 or len(set(names))!=100 or any("/" in n for n in names) or bad is not None:
 raise SystemExit("final archive audit failed")
p={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),"status":"COMPLETE",
   "decision":"READY_FOR_MANUAL_CODABENCH_UPLOAD","recipe":{"top_k":1,"fold":0,"tta":False,
   "min_seed_coverage":0.25,"min_refined_overlap":0.25,"min_volume_ratio":0.25,"max_volume_ratio":4.0},
   "anchor":means(a),"candidate":means(c),"public_inference_seconds":ps,"hidden_inference_seconds":hs,
   "public_replacement_audit":str(pa.resolve()),"public_replacement_audit_sha256":sha(pa),
   "hidden_replacement_audit":str(ha.resolve()),"hidden_replacement_audit_sha256":sha(ha),
   "smoke_terminal":str(s.resolve()),"smoke_terminal_sha256":sha(s),
   "champion_manifest":str(h.resolve()),"champion_manifest_sha256":sha(h),
   "archive":str(z.resolve()),"archive_sha256":sha(z),"archive_members":100,
   "archive_flat":True,"archive_crc_valid":True,"uploaded":False}
tmp=out.with_name(f".{out.name}.{os.getpid()}.tmp");tmp.write_text(json.dumps(p,indent=2)+"\n");os.replace(tmp,out)
print(json.dumps(p,indent=2))
PY
