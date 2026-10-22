#!/usr/bin/env bash
set -euo pipefail

# Two-GPU deadline continuation. Reuses the fully audited prompt inventories
# produced by run_gfconf_top1_submission_20260824.sh, gates on all public cases,
# then applies the identical label-free recipe to hidden cases. Never uploads.
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
sharded="${output}/sharded_v2"
terminal="${output}/TERMINAL.json"
archive="${workspace}/submissions/gf_ll_top1_cbg41.zip"
lock="${output}/controller.lock"

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "gf_conf top-1 full controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

for required in "${champion}" "${smoke_terminal}" "${public_images}" \
  "${public_labels}" "${hidden_images}" "${output}/public_prompt_audit.json" \
  "${output}/hidden_prompt_audit.json" \
  "${model_root}/LesionLocatorSeg/bbox_optimized/fold_0/checkpoint_final.pth"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${smoke_terminal}" "${output}/public_prompt_audit.json" \
  "${output}/hidden_prompt_audit.json" <<'PY'
import json,sys
s,p,h=(json.load(open(x,encoding="utf-8")) for x in sys.argv[1:])
if s["decision"]!="QUALIFY_FOR_FULL_PUBLIC":raise SystemExit("smoke did not qualify")
if p["cases"]!=50 or p["selected_components"]!=50:raise SystemExit("public prompt audit mismatch")
if h["cases"]!=100 or h["selected_components"]!=100:raise SystemExit("hidden prompt audit mismatch")
PY

mapfile -t anchor_paths < <("${python_bin}" - "${champion}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"));pub=Path(p["public_predictions"]);hid=Path(p["hidden_predictions"])
m=Path(p["public_metrics"]);z=Path(p["archive"])
if hashlib.sha256(m.read_bytes()).hexdigest()!=p["public_metrics_sha256"]:raise SystemExit("metric identity mismatch")
if hashlib.sha256(z.read_bytes()).hexdigest()!=p["archive_sha256"]:raise SystemExit("archive identity mismatch")
if not pub.is_dir() or not hid.is_dir():raise SystemExit("champion prediction assets missing")
print(pub.resolve());print(hid.resolve())
PY
)
[[ "${#anchor_paths[@]}" -eq 2 ]] || exit 2
public_anchor="${anchor_paths[0]}"; hidden_anchor="${anchor_paths[1]}"

make_shards() {
  local images="$1" prompts="$2" root="$3" expected="$4"
  "${python_bin}" - "${images}" "${prompts}" "${root}" "${expected}" <<'PY'
import os,sys
from pathlib import Path
images,prompts,root=map(Path,sys.argv[1:4]);expected=int(sys.argv[4])
files=sorted(images.glob("*_0000.nii.gz"))
if len(files)!=expected:raise SystemExit(f"expected {expected} images, found {len(files)}")
for shard in range(2):
 (root/f"shard{shard}"/"inputs").mkdir(parents=True,exist_ok=True)
 (root/f"shard{shard}"/"prompts").mkdir(parents=True,exist_ok=True)
 (root/f"shard{shard}"/"instances").mkdir(parents=True,exist_ok=True)
for index,image in enumerate(files):
 shard=root/f"shard{index%2}"
 prompt=prompts/image.name.replace("_0000.nii.gz",".nii.gz")
 if not prompt.is_file():raise SystemExit(f"missing prompt {prompt}")
 for source,destination in ((image,shard/"inputs"/image.name),(prompt,shard/"prompts"/prompt.name)):
  if destination.is_symlink():
   if destination.resolve()!=source.resolve():raise SystemExit(f"stale shard link {destination}")
  elif destination.exists():raise SystemExit(f"unexpected shard asset {destination}")
  else:os.symlink(source.resolve(),destination)
for shard in range(2):
 if len(list((root/f"shard{shard}"/"inputs").glob("*.nii.gz")))!=expected//2:
  raise SystemExit("balanced shard inventory mismatch")
PY
}

mkdir -p "${sharded}" "${output}/metrics"
make_shards "${public_images}" "${output}/public_prompts" "${sharded}/public" 50
make_shards "${hidden_images}" "${output}/hidden_prompts" "${sharded}/hidden" 100

export MKL_THREADING_LAYER=GNU
run_pair() {
  local kind="$1"
  local start
  start=$(date +%s)
  for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${script_dir}" "${python_bin}" -u \
      -c 'from lesionlocator.inference.lesionlocator_segment import predict_seg_from_prompt; predict_seg_from_prompt()' \
      -i "${sharded}/${kind}/shard${gpu}/inputs" \
      -p "${sharded}/${kind}/shard${gpu}/prompts" -t box \
      -o "${sharded}/${kind}/shard${gpu}/instances" -m "${model_root}" -f 0 \
      --disable_tta --disable_progress_bar -npp 2 -nps 2 \
      > "${sharded}/${kind}/shard${gpu}/inference.log" 2>&1 &
    pair_pids[${gpu}]=$!
  done
  wait "${pair_pids[0]}"
  wait "${pair_pids[1]}"
  echo $(($(date +%s)-start))
}

merge_instances() {
  local kind="$1" expected="$2" merged="${sharded}/${kind}/instances_merged"
  mkdir -p "${merged}"
  "${python_bin}" - "${sharded}/${kind}" "${merged}" "${expected}" <<'PY'
import os,sys
from pathlib import Path
root,merged=map(Path,sys.argv[1:3]);expected=int(sys.argv[3]);files=[]
for shard in range(2):files.extend(sorted((root/f"shard{shard}"/"instances").glob("*_lesion_*.nii.gz")))
if len(files)!=expected or len({p.name for p in files})!=expected:raise SystemExit(f"instance inventory mismatch: {len(files)}")
for source in files:
 destination=merged/source.name
 if destination.is_symlink():
  if destination.resolve()!=source.resolve():raise SystemExit(f"stale merged link {destination}")
 elif destination.exists():raise SystemExit(f"unexpected merged asset {destination}")
 else:os.symlink(source.resolve(),destination)
PY
}

public_seconds=$(run_pair public)
merge_instances public 50
"${python_bin}" -u "${script_dir}/materialize_topk_prompt_replacement.py" \
  --references "${public_images}" --instances "${sharded}/public/instances_merged" \
  --anchor "${public_anchor}" --prompts "${output}/public_prompts" \
  --output "${output}/public_predictions" --audit "${output}/public_replacement_audit.json" \
  --expected-count 50 --min-seed-coverage 0.25 --min-refined-overlap 0.25 \
  --min-volume-ratio 0.25 --max-volume-ratio 4.0 > "${output}/public_materialize.log" 2>&1
"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${public_anchor}" --gt "${public_labels}" --csv "${output}/metrics/anchor.csv" \
  --workers 4 > "${output}/metrics/anchor.log" 2>&1
"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${output}/public_predictions" --gt "${public_labels}" --csv "${output}/metrics/candidate.csv" \
  --workers 4 > "${output}/metrics/candidate.log" 2>&1

public_gate=$("${python_bin}" - "${output}/metrics/anchor.csv" "${output}/metrics/candidate.csv" <<'PY'
import csv,sys
def mean(path,key):
 with open(path,newline="") as f:r=list(csv.DictReader(f))
 if len(r)!=50:raise SystemExit("50-case coverage required")
 return sum(float(x[key]) for x in r)/50
ad,an=mean(sys.argv[1],"Lesion_DSC"),mean(sys.argv[1],"Lesion_NSD")
cd,cn=mean(sys.argv[2],"Lesion_DSC"),mean(sys.argv[2],"Lesion_NSD")
print("PROMOTE" if cd>=ad+0.0005 and cn>=an-0.002 else "PRUNE")
print(f"anchor={ad:.6f}/{an:.6f} candidate={cd:.6f}/{cn:.6f}",file=sys.stderr)
PY
)
if [[ "${public_gate}" != "PROMOTE" ]]; then
  "${python_bin}" - "${output}/metrics/anchor.csv" "${output}/metrics/candidate.csv" "${terminal}" <<'PY'
import csv,json,os,sys
from datetime import datetime
from pathlib import Path
def means(p):
 with open(p,newline="") as f:r=list(csv.DictReader(f))
 return {k:sum(float(x[k]) for x in r)/len(r) for k in ("Lesion_DSC","Lesion_NSD")}
out=Path(sys.argv[3]);p={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
"status":"COMPLETE","decision":"PRUNE_AFTER_FULL_PUBLIC","anchor":means(sys.argv[1]),
"candidate":means(sys.argv[2]),"submission_created":False,"uploaded":False}
t=out.with_name(f".{out.name}.{os.getpid()}.tmp");t.write_text(json.dumps(p,indent=2)+"\n");os.replace(t,out)
PY
  exit 0
fi

hidden_seconds=$(run_pair hidden)
merge_instances hidden 100
"${python_bin}" -u "${script_dir}/materialize_topk_prompt_replacement.py" \
  --references "${hidden_images}" --instances "${sharded}/hidden/instances_merged" \
  --anchor "${hidden_anchor}" --prompts "${output}/hidden_prompts" \
  --output "${output}/hidden_predictions" --audit "${output}/hidden_replacement_audit.json" \
  --expected-count 100 --min-seed-coverage 0.25 --min-refined-overlap 0.25 \
  --min-volume-ratio 0.25 --max-volume-ratio 4.0 > "${output}/hidden_materialize.log" 2>&1
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
with zipfile.ZipFile(z) as q:names=q.namelist();bad=q.testzip()
if len(names)!=100 or len(set(names))!=100 or any("/" in n for n in names) or bad is not None:raise SystemExit("archive audit failed")
p={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),"status":"COMPLETE",
"decision":"READY_FOR_MANUAL_CODABENCH_UPLOAD","execution":"two_gpu_sharded",
"recipe":{"top_k":1,"fold":0,"tta":False,"min_seed_coverage":0.25,"min_refined_overlap":0.25,
"min_volume_ratio":0.25,"max_volume_ratio":4.0},"anchor":means(a),"candidate":means(c),
"public_inference_seconds":ps,"hidden_inference_seconds":hs,"public_replacement_audit":str(pa.resolve()),
"public_replacement_audit_sha256":sha(pa),"hidden_replacement_audit":str(ha.resolve()),
"hidden_replacement_audit_sha256":sha(ha),"smoke_terminal":str(s.resolve()),"smoke_terminal_sha256":sha(s),
"champion_manifest":str(h.resolve()),"champion_manifest_sha256":sha(h),"archive":str(z.resolve()),
"archive_sha256":sha(z),"archive_members":100,"archive_flat":True,"archive_crc_valid":True,"uploaded":False}
t=out.with_name(f".{out.name}.{os.getpid()}.tmp");t.write_text(json.dumps(p,indent=2)+"\n");os.replace(t,out);print(json.dumps(p,indent=2))
PY
