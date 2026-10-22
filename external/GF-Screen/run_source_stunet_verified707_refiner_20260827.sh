#!/usr/bin/env bash
set -euo pipefail

# Five-fold, case-held-out classifier for source-stratified STU proposals on
# the exact 707 verified-complete training cohort. The STU contour is preserved
# at deployment (`seed` mode); this experiment learns only whether a proposed
# component/tile is safe to add. No public/hidden inference or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
proposal_terminal="${project}/external/STU-Net/pancancer/analysis/source_stunet_verified707_proposals_20260827/PROPOSALS_COMPLETE.json"
proposals="${project}/external/STU-Net/pancancer/analysis/source_stunet_verified707_proposals_20260827/predictions"
work="${root}/analysis/source_stunet_verified707_refiner_20260827"
patches="${work}/patches_bbox24_source_stunet"
audit="${patches}/AUDIT.json"
oof="${work}/five_fold_oof_summary_classification.json"
gate="${work}/PATCH_OOF_GATE.json"
terminal="${work}/TRAINING_TERMINAL.json"
selector="${root}/select_stunet_proposal_refiner_oof.py"
label_correction="${root}/analysis/fpr_training_mining_707_20260822/LABEL_CONTRACT_CORRECTION_20260825.json"

fold_dir() { printf '%s/runs/candidate_refiner_source_stunet_b32_fold%s_20260827' "${root}" "$1"; }

mkdir -p "${work}" "${root}/runs"
exec 9>"${root}/runs/run_source_stunet_verified707_refiner_20260827.lock"
flock -n 9 || { echo "source-STU verified-707 refiner already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected source-STU refiner terminal: ${status}" >&2; exit 2 ;;
  esac
fi

# Direct-launch guard: annotation-negative proposals are unknown because the
# challenge-distributed masks may omit non-primary lesions. A precision
# classifier requiring negative examples cannot be trained from this cohort.
"${python_bin}" - "${label_correction}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
correction,output=map(Path,sys.argv[1:]); p=json.load(open(correction,encoding="utf-8"))
if (p.get("status")!="INVALIDATE_COMPLETE_LABEL_INTERPRETATION"
    or "authoritative complete-label negative mining" not in p.get("prohibited_use",[])):
 raise SystemExit("missing authoritative 707 label-contract correction")
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES",
 "reason":"annotation-negative STU proposals are unknown under info.txt",
 "label_contract_correction":{"path":str(correction.resolve()),"sha256":sha(correction)},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
exit 0

write_failure() {
  local reason="$1"
  "${python_bin}" - "${reason}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
reason=sys.argv[1]; output=Path(sys.argv[2]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"RETRY_FROM_VALIDATED_PATCH_OR_OPTIMIZER_CACHE",
 "reason":reason,"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_failure "source-STU refiner failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

while [[ ! -s "${proposal_terminal}" ]]; do
  printf '%s waiting for exact-707 source-STU mask proposals\n' "$(date -Ins)"
  sleep 30
done
for file in "${manifest}" "${proposal_terminal}" "${selector}"; do
  [[ -s "${file}" ]] || { echo "missing source-STU refiner input: ${file}" >&2; exit 2; }
done
"${python_bin}" - "${manifest}" "${proposal_terminal}" "${proposals}" <<'PY'
import hashlib,json,sys
from pathlib import Path
manifest,terminal,proposals=map(Path,sys.argv[1:]); m=json.load(open(manifest,encoding="utf-8")); p=json.load(open(terminal,encoding="utf-8"))
names={str(item["prediction_name"]) for item in m.get("items",[])}
observed={path.name for path in proposals.glob("*.nii.gz")}
if (len(names)!=707 or any(item.get("complete_label") is not True for item in m.get("items",[]))
    or p.get("status")!="COMPLETE" or p.get("decision")!="TRAIN_STU_PROPOSAL_PRECISION_GATE"
    or p.get("cases")!=707 or p.get("proposal_threshold")!=0.5
    or p.get("probabilities_saved") or observed!=names
    or p.get("training_labels_read") or p.get("public_read") or p.get("hidden_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit("invalid source-STU proposal terminal/coverage")
record=p.get("manifest",{})
if record.get("path")!=str(manifest.resolve()) or record.get("sha256")!=hashlib.sha256(manifest.read_bytes()).hexdigest():
    raise SystemExit("proposal terminal manifest provenance mismatch")
PY

export MPLCONFIGDIR="${work}/matplotlib"
export TMPDIR="/tmp/gfref_source_stunet_20260827"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
if [[ ! -s "${patches}/COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/prepare_candidate_refiner_patches.py" \
    --manifest "${manifest}" --proposal "source_stunet=${proposals}" \
    --component-policy source_separate --compact-source-components \
    --output "${patches}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 \
    --max-negatives-per-case 64 --max-negative-tiles 1 --workers 5 --resume \
    >"${work}/prepare_patches.log" 2>&1
fi
if [[ ! -s "${audit}" ]]; then
  "${python_bin}" -u "${root}/audit_candidate_refiner_patches.py" \
    --index "${patches}/index.csv" --summary "${patches}/summary.json" \
    --output "${audit}" --workers 8 >"${work}/audit_patches.log" 2>&1
fi
"${python_bin}" - "${manifest}" "${proposal_terminal}" "${patches}/summary.json" \
  "${patches}/index.csv" "${audit}" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
manifest,proposal,summary_path,index,audit_path=map(Path,sys.argv[1:]); summary=json.load(open(summary_path)); audit=json.load(open(audit_path))
with index.open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
checks={
 "manifest":summary.get("manifest",{}).get("sha256")==hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "proposal_terminal":json.load(open(proposal,encoding="utf-8")).get("cases")==707,
 "cases":summary.get("cases")==707 and len({row["case"] for row in rows})==707,
 "source":summary.get("proposal_sources")==[{"name":"source_stunet","directory":str(Path(summary["proposal_sources"][0]["directory"]).resolve())}]
          and {row["source_name"] for row in rows}=={"source_stunet"},
 "geometry":summary.get("shape")==[64,64,64] and summary.get("spacing")==[1.5,1.5,1.5]
            and summary.get("bbox_margin_mm")==24.0 and summary.get("tile_stride_mm")==72.0,
 "policy":summary.get("component_policy")=="source_separate" and summary.get("compact_source_components") is True,
 "labels":summary.get("positive",0)>0 and summary.get("negative",0)>0,
 "audit_hash":audit.get("index_sha256")==hashlib.sha256(index.read_bytes()).hexdigest(),
 "audit":audit.get("patches")==len(rows) and audit.get("exact_file_coverage") is True
         and audit.get("all_prompts_nonempty") is True and audit.get("all_values_finite") is True
         and audit.get("all_targets_binary_and_label_consistent") is True,
}
failed=[name for name,value in checks.items() if not value]
if failed: raise SystemExit(f"invalid source-STU patch cache: {failed}")
PY

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
train_fold() {
  local gpu="$1" fold="$2" output
  output="$(fold_dir "${fold}")"; mkdir -p "${output}"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best_classification.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
    --index "${patches}/index.csv" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
    --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260827 \
    --segmentation-threshold-max 0.95 --validation-interval 3 \
    --validation-batch-size 32 --wandb-project pancancer-ct \
    --wandb-name "candidate_refiner_source_stunet_b32_fold${fold}" --resume \
    >"${output}/train.log" 2>&1
}

remaining=(0 1 2 3 4)
job_pid=("" ""); job_fold=("" "")
while [[ ${#remaining[@]} -gt 0 || -n "${job_pid[0]}" || -n "${job_pid[1]}" ]]; do
  for gpu in 0 1; do
    pid="${job_pid[${gpu}]}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      status=0; wait "${pid}" || status=$?
      [[ "${status}" -eq 0 ]] || exit "${status}"
      job_pid[${gpu}]=""; job_fold[${gpu}]=""
    fi
  done
  if [[ ${#remaining[@]} -gt 0 ]]; then
    for gpu in 0 1; do
      [[ ${#remaining[@]} -gt 0 ]] || break
      [[ -z "${job_pid[${gpu}]}" ]] || continue
      used=$(gpu_memory_mib "${gpu}")
      if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]]; then
        fold="${remaining[0]}"; remaining=("${remaining[@]:1}")
        train_fold "${gpu}" "${fold}" & job_pid[${gpu}]=$!; job_fold[${gpu}]="${fold}"
        printf '%s dispatched source-STU refiner fold %s to GPU %s\n' "$(date -Ins)" "${fold}" "${gpu}"
      fi
    done
  fi
  [[ ${#remaining[@]} -eq 0 && -z "${job_pid[0]}" && -z "${job_pid[1]}" ]] || sleep 30
done

oof_args=()
for fold in 0 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  oof_args+=(--checkpoint "${output}/checkpoint_best_classification.pt")
  oof_args+=(--prediction "${output}/validation_predictions_best_classification.npz")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${oof_args[@]}" --output "${oof}" \
  --bootstrap 1000 --seed 20260827 >"${work}/summarize_oof.log" 2>&1
"${python_bin}" -u "${selector}" --index "${patches}/index.csv" \
  --summary "${oof}" --output "${gate}" --bootstrap 5000 --seed 20260827 \
  >"${work}/select_patch_gate.log" 2>&1

"${python_bin}" - "${manifest}" "${proposal_terminal}" "${patches}/summary.json" \
  "${audit}" "${oof}" "${gate}" "${selector}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
manifest,proposal,patch_summary,audit,oof,gate,selector,output=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); decision=json.load(open(gate,encoding="utf-8"))["decision"]
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,"cases":707,"folds":5,
 "architecture":"3D residual U-Net candidate confirmer, base_channels=32 (~9.49M parameters)",
 "deployment_role":"classify source-STU components/tiles; preserve accepted STU seed contour",
 "training":{"epochs":30,"batch_size_per_gpu":8,"optimizer":"AdamW","learning_rate":3e-4,
   "weight_decay":1e-4,"scheduler":"5-epoch linear warmup then cosine decay",
   "sampling_temperature":0.5,"samples_per_epoch":8192,"grouped_case_folds":5},
 "patch_geometry":{"shape":[64,64,64],"spacing_mm":[1.5,1.5,1.5],
   "bbox_margin_mm":24,"tile_stride_mm":72,"negative_cap_per_case":64},
 "assets":{name:{"path":str(path.resolve()),"sha256":sha(path)} for name,path in {
   "manifest":manifest,"proposal_terminal":proposal,"patch_summary":patch_summary,
   "patch_audit":audit,"patch_oof":oof,"patch_gate":gate,"selector":selector}.items()},
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
trap - EXIT
