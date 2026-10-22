#!/usr/bin/env bash
set -euo pipefail

# Leakage-free full-volume OOF for the source-STU component confirmer. Every
# case is inferred by its excluded-fold classifier. Accepted STU seed contours
# are added to a frozen current-GF training anchor, then scored against the 707
# verified-complete labels. No public/hidden inference, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/source_stunet_verified707_refiner_20260827"
training_terminal="${train}/TRAINING_TERMINAL.json"
patch_gate="${train}/PATCH_OOF_GATE.json"
index="${train}/patches_bbox24_source_stunet/index.csv"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
source_stunet="${root}/../STU-Net/pancancer/analysis/source_stunet_verified707_proposals_20260827/predictions"
currentgf="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
output="${root}/analysis/source_stunet_verified707_fullvolume_oof_20260827"
staging="${output}/staging"
combined="${output}/candidate_combined"
anchor_metrics="${output}/currentgf_anchor.metrics.csv"
candidate_metrics="${output}/source_stunet_confirmed_union.metrics.csv"
robust="${output}/paired_vs_currentgf.robust.json"
terminal="${output}/FULL_VOLUME_OOF_TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_source_stunet_b32_fold%s_20260827' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }
fold_masks() { printf '%s/fold%s/masks' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_source_stunet_verified707_fullvolume_oof_20260827.lock"
flock -n 9 || { echo "source-STU verified-707 full-volume OOF already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE || "${status}" == SKIPPED ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${training_terminal}" \
    "${patch_gate}" "${anchor_metrics}" "${candidate_metrics}" "${robust}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]; paths=list(map(Path,sys.argv[4:-1])); output=Path(sys.argv[-1]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
gate=json.load(open(paths[1],encoding="utf-8")) if paths[1].is_file() else {}
robust=json.load(open(paths[4],encoding="utf-8")) if paths[4].is_file() else {}
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,"cases":707,
 "deployment":"frozen current-GF anchor plus excluded-fold-confirmed source-STU seed contours",
 "classification_threshold":gate.get("classification_threshold"),
 "patch_gate":asset(paths[1]),"training":asset(paths[0]),
 "anchor_metrics":asset(paths[2]),"candidate_metrics":asset(paths[3]),
 "paired_gate":asset(paths[4]),"paired_gate_payload":robust,
 "labels_read":"verified-complete training labels after excluded-fold inference only",
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED RETRY_FROM_VALIDATED_OOF_CACHE "failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

while [[ ! -s "${training_terminal}" ]]; do
  printf '%s waiting for source-STU five-fold training terminal\n' "$(date -Ins)"
  sleep 30
done
training_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${training_terminal}")
if [[ "${training_decision}" != QUALIFY_STUNET_SEED_FULL_VOLUME_OOF ]]; then
  [[ "${training_decision}" == REJECT_STUNET_SEED_PATCH_GATE ]]
  write_terminal SKIPPED REJECT_STUNET_SEED_PATCH_GATE "high-precision grouped patch OOF gate failed"
  trap - EXIT; exit 0
fi
for required in "${patch_gate}" "${index}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing source-STU OOF input: ${required}" >&2; exit 2; }
done
classification_threshold=$("${python_bin}" - "${training_terminal}" "${patch_gate}" <<'PY'
import hashlib,json,sys
from pathlib import Path
training,gate_path=map(Path,sys.argv[1:]); t=json.load(open(training,encoding="utf-8")); g=json.load(open(gate_path,encoding="utf-8"))
record=t.get("assets",{}).get("patch_gate",{})
if (t.get("status")!="COMPLETE" or t.get("decision")!="QUALIFY_STUNET_SEED_FULL_VOLUME_OOF"
    or record.get("path")!=str(gate_path.resolve()) or record.get("sha256")!=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    or g.get("status")!="COMPLETE" or g.get("deployment_mode")!="seed"
    or not all(g.get("checks",{}).values()) or g.get("public_read") or g.get("hidden_read")):
    raise SystemExit("invalid source-STU patch qualification")
value=float(g["classification_threshold"])
if value not in [round(0.50+0.05*i,2) for i in range(10)]: raise SystemExit("threshold outside frozen grid")
print(f"{value:.2f}")
PY
)

if [[ ! -s "${staging}/STAGING_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
    --index "${index}" --manifest "${manifest}" \
    --proposal "source_stunet=${source_stunet}" --proposal "currentgf=${currentgf}" \
    --output "${staging}" --folds 5 --seed 20260827 >"${output}/stage.log" 2>&1
fi
"${python_bin}" - "${staging}/STAGING_COMPLETE.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("cases")!=707 or p.get("proposal_layout")!="named_sources"
    or set(p.get("proposal_sources",{}))!={"source_stunet","currentgf"}
    or not p.get("exact_case_coverage_once") or not p.get("exact_proposal_coverage")):
    raise SystemExit("invalid source-STU OOF staging")
PY

prepare_fold() {
  local fold="$1" stage="${staging}/fold${fold}" work
  work="$(fold_work "${fold}")"; mkdir -p "${work}"
  [[ -s "${work}/PREPARE_COMPLETE.txt" ]] && return
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${stage}/images" --proposal "source_stunet=${stage}/proposals/source_stunet" \
    --component-policy source_separate --base "${stage}/proposals/currentgf" --work "${work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 --resume >"${output}/fold${fold}.prepare.log" 2>&1
}
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do prepare_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]]
done

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1); free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for source-STU OOF GPU: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
infer_fold() {
  local gpu="$1" fold="$2" work checkpoint
  work="$(fold_work "${fold}")"; checkpoint="$(fold_dir "${fold}")/checkpoint_best_classification.pt"
  if ! "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
      --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${checkpoint}" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
      --work "${work}" --checkpoint "${checkpoint}" --batch-size 32 --workers 2 --resume \
      >"${output}/fold${fold}.infer.log" 2>&1
  fi
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${checkpoint}" >/dev/null
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    --require-predictions --require-inference-plan >"${output}/fold${fold}.audit.log" 2>&1
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_sequence 0 0 2 4 & p0=$!; infer_sequence 1 1 3 & p1=$!; status=0
  wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]
elif [[ "${free0}" -eq 1 ]]; then infer_sequence 0 0 1 2 3 4
else infer_sequence 1 0 1 2 3 4
fi

materialize_fold() {
  local fold="$1" stage="${staging}/fold${fold}" work masks
  work="$(fold_work "${fold}")"; masks="$(fold_masks "${fold}")"; mkdir -p "${masks}"
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${stage}/proposals/currentgf" --output "${masks}" --resume \
    --classification-threshold "${classification_threshold}" --segmentation-threshold 0.50 \
    --mode refined --source-mode source_stunet=seed \
    --min-seed-coverage 0 --min-refined-overlap 0 --min-volume-ratio 0 \
    --max-volume-ratio 1000000000000 >"${output}/fold${fold}.materialize.log" 2>&1
}
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do materialize_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]]
done

mkdir -p "${combined}"
"${python_bin}" - "${output}" "${combined}" "${staging}/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); seen=set()
for fold in range(5):
 for source in (root/f"fold{fold}"/"masks").glob("*.nii.gz"):
  if source.name in seen: raise SystemExit(f"duplicate OOF mask: {source.name}")
  target=combined/source.name
  if target.is_symlink() or target.exists():
   if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit("OOF mask collision")
  else: target.symlink_to(source.resolve(strict=True))
  seen.add(source.name)
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=707 or seen!=expected: raise SystemExit(f"OOF mask coverage {len(seen)}/{len(expected)}")
PY
"${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${staging}/all/proposals/currentgf" \
  --gt "${staging}/all/labels" --csv "${anchor_metrics}" --workers 6 >"${output}/anchor.score.log" 2>&1
"${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${combined}" \
  --gt "${staging}/all/labels" --csv "${candidate_metrics}" --workers 6 >"${output}/candidate.score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name currentgf_fixed_anchor --candidate-name source_stunet_confirmed_seed_union \
  --min-dsc-gain 0.015 --min-nsd-gain 0.0 --min-win-rate 0.52 \
  --max-catastrophic-rate 0.10 --bootstrap 20000 --seed 20260827 \
  --output "${robust}" >"${output}/robust.log" 2>&1
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
if [[ "${decision}" == PROMOTE ]]; then
  write_terminal COMPLETE QUALIFY_STUNET_SEED_FIXED_PUBLIC "passed +0.015 DSC/non-negative NSD exact-707 full-volume OOF gate"
else
  write_terminal COMPLETE REJECT_STUNET_SEED_FULL_VOLUME_OOF "did not materially improve the frozen current-GF anchor"
fi
trap - EXIT
