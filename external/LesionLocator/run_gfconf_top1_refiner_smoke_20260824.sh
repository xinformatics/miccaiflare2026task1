#!/usr/bin/env bash
set -euo pipefail

# Fixed five-case public smoke of a runtime-bounded LesionLocator boundary
# refinement. Only the largest retained gf_conf component is prompted per case;
# all unprompted anchor components remain immutable. No hidden inference,
# packaging, submission, or upload path exists.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
champion="${workspace}/external/GF-Screen/analysis/DEPLOYMENT_CHAMPION.json"
images="${workspace}/validation/Validation-Public-Images"
labels="${workspace}/validation/Validation-Public-Labels"
model_root="${script_dir}/LesionLocatorCheckpoint"
output="${script_dir}/public_audit/gfconf_top1_fold0_notta_smoke_20260824"
terminal="${output}/SMOKE_TERMINAL.json"
lock="${output}/controller.lock"

mkdir -p "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "gf_conf top-1 LesionLocator smoke already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

for required in "${champion}" "${images}" "${labels}" \
  "${model_root}/LesionLocatorSeg/bbox_optimized/fold_0/checkpoint_final.pth"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
anchor=$("${python_bin}" - "${champion}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8")); anchor=Path(p["public_predictions"])
metrics=Path(p["public_metrics"])
if not anchor.is_dir() or not metrics.is_file(): raise SystemExit("champion assets missing")
if hashlib.sha256(metrics.read_bytes()).hexdigest()!=p["public_metrics_sha256"]:
 raise SystemExit("champion metric identity mismatch")
print(anchor.resolve())
PY
)

mkdir -p "${output}/inputs" "${output}/labels" "${output}/anchor" "${output}/prompts" \
  "${output}/instances" "${output}/predictions" "${output}/metrics"
cases=(0001 0005 0013 0073 0099)
for case_id in "${cases[@]}"; do
  ln -sfn "${images}/FLARE23Ts_${case_id}_0000.nii.gz" \
    "${output}/inputs/FLARE23Ts_${case_id}_0000.nii.gz"
  ln -sfn "${anchor}/FLARE23Ts_${case_id}.nii.gz" \
    "${output}/anchor/FLARE23Ts_${case_id}.nii.gz"
  ln -sfn "${labels}/FLARE23Ts_${case_id}.nii.gz" \
    "${output}/labels/FLARE23Ts_${case_id}.nii.gz"
done

"${python_bin}" -u "${script_dir}/prepare_topk_component_prompts.py" \
  --input "${output}/anchor" --output "${output}/prompts" \
  --audit "${output}/prompt_audit.json" --top-k 1 --min-mm3 0 \
  --max-mm3 inf --expected-count 5 > "${output}/prepare.log" 2>&1
selected=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selected_components"])' \
  "${output}/prompt_audit.json")
[[ "${selected}" -eq 5 ]] || { echo "expected five nonempty top-1 prompts" >&2; exit 2; }

while :; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "${#used[@]}" -eq 2 && "${used[0]}" =~ ^[0-9]+$ && "${used[1]}" =~ ^[0-9]+$ ]] || exit 2
  # Deadline-aware smoke may share a 48-GiB GPU with the small 2-D trainer.
  # The 16-GiB ceiling still leaves at least 32 GiB for LesionLocator.
  if [[ "${used[0]}" -le 16384 || "${used[1]}" -le 16384 ]]; then
    if [[ "${used[0]}" -le "${used[1]}" ]]; then gpu=0; else gpu=1; fi
    break
  fi
  printf '%s waiting for an idle GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used[0]}" "${used[1]}"
  sleep 30
done

export MKL_THREADING_LAYER=GNU
start_seconds=$(date +%s)
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${script_dir}" "${python_bin}" -u \
  -c 'from lesionlocator.inference.lesionlocator_segment import predict_seg_from_prompt; predict_seg_from_prompt()' \
  -i "${output}/inputs" -p "${output}/prompts" -t box \
  -o "${output}/instances" -m "${model_root}" -f 0 \
  --disable_tta --disable_progress_bar -npp 2 -nps 2 \
  > "${output}/inference.log" 2>&1
end_seconds=$(date +%s)
runtime=$((end_seconds - start_seconds))
printf '%s\n' "${runtime}" > "${output}/inference_seconds.txt"

"${python_bin}" -u "${script_dir}/materialize_topk_prompt_replacement.py" \
  --references "${output}/inputs" --instances "${output}/instances" \
  --anchor "${output}/anchor" --prompts "${output}/prompts" \
  --output "${output}/predictions" --audit "${output}/replacement_audit.json" \
  --expected-count 5 --min-seed-coverage 0.25 --min-refined-overlap 0.25 \
  --min-volume-ratio 0.25 --max-volume-ratio 4.0 \
  > "${output}/materialize.log" 2>&1

"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${output}/anchor" --gt "${output}/labels" \
  --csv "${output}/metrics/anchor.csv" --workers 4 \
  > "${output}/metrics/anchor.log" 2>&1
"${python_bin}" -u "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${output}/predictions" --gt "${output}/labels" \
  --csv "${output}/metrics/candidate.csv" --workers 4 \
  > "${output}/metrics/candidate.log" 2>&1

"${python_bin}" - "${output}/metrics/anchor.csv" \
  "${output}/metrics/candidate.csv" "${output}/replacement_audit.json" \
  "${runtime}" "${champion}" "${terminal}" <<'PY'
import csv,hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
anchor_path,candidate_path,audit_path=map(Path,sys.argv[1:4])
runtime=int(sys.argv[4]); champion=Path(sys.argv[5]); output=Path(sys.argv[6])
def read(path):
 with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
 if len(rows)!=5: raise SystemExit(f"five-case metric coverage required: {path}")
 return {r["Name"]:r for r in rows}
anchor=read(anchor_path); candidate=read(candidate_path)
if set(anchor)!=set(candidate): raise SystemExit("paired metric case mismatch")
mean=lambda rows,key:sum(float(row[key]) for row in rows.values())/len(rows)
ad,an=mean(anchor,"Lesion_DSC"),mean(anchor,"Lesion_NSD")
cd,cn=mean(candidate,"Lesion_DSC"),mean(candidate,"Lesion_NSD")
changed=sum(anchor[name].get("pred_fg_voxels")!=candidate[name].get("pred_fg_voxels") for name in anchor)
audit=json.loads(audit_path.read_text(encoding="utf-8"))
seconds_per_prompt=runtime/5.0
checks={
 "dsc_within_0p03":cd>=ad-0.03,
 "nsd_within_0p03":cn>=an-0.03,
 "runtime_at_most_120_seconds_per_prompt":seconds_per_prompt<=120,
 "at_least_one_output_changed":changed>0,
 "at_least_one_refinement_accepted":int(audit["accepted_cases"])>0,
}
decision="QUALIFY_FOR_FULL_PUBLIC" if all(checks.values()) else "KEEP_GFCONF"
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,"checks":checks,"cases":5,
 "runtime_seconds":runtime,"seconds_per_prompt":seconds_per_prompt,
 "anchor":{"Lesion_DSC":ad,"Lesion_NSD":an,"metrics":str(anchor_path.resolve()),"sha256":sha(anchor_path)},
 "candidate":{"Lesion_DSC":cd,"Lesion_NSD":cn,"metrics":str(candidate_path.resolve()),"sha256":sha(candidate_path)},
 "delta":{"Lesion_DSC":cd-ad,"Lesion_NSD":cn-an},"changed_cases":changed,
 "replacement_audit":str(audit_path.resolve()),"replacement_audit_sha256":sha(audit_path),
 "champion_manifest":str(champion.resolve()),"champion_manifest_sha256":sha(champion),
 "selection_basis":"fixed five-case runtime/quality smoke only; not a deployment promotion",
 "public_labels_read":True,"hidden_data_read":False,"submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
