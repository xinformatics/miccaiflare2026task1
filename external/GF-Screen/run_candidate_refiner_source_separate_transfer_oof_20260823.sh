#!/usr/bin/env bash
set -euo pipefail

# Held-out transfer diagnostic for compact source-specific components.  Every
# training case is inferred only by its excluded grouped-fold checkpoint.  The
# script has no public/hidden data, packaging, submission, or upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v2_training_20260822"
training_complete="${train}/TRAINING_COMPLETE.txt"
selection="${train}/checkpoint_selection.json"
patch_oof="${train}/five_fold_oof_summary.json"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
prewarm_root="${root}/analysis/candidate_refiner_v3_source_separate_oof_20260823"
prewarm="${prewarm_root}/PREWARM_COMPLETE.json"
union_root="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
union_complete="${union_root}/FULL_VOLUME_OOF_COMPLETE.txt"
union_calibration="${union_root}/deployment_calibration.json"
union_staging="${union_root}/staging"
primary_saved="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
multisource_saved="${root}/public_sweeps/candidate_refiner_v2_multisource_saved_20260823/SAVE_ONLY_COMPLETE.txt"
output="${root}/analysis/candidate_refiner_v3_source_separate_transfer_oof_20260823"
calibration="${output}/deployment_calibration.json"
robust="${output}/paired_vs_union.json"
complete="${output}/TRANSFER_OOF_COMPLETE.json"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${prewarm_root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_source_separate_transfer_oof_20260823.lock"
flock -n 9 || { echo "source-separated transfer OOF already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${training_complete}" || ! -s "${prewarm}" \
      || ! -s "${union_complete}" || ! -s "${primary_saved}" \
      || ! -s "${multisource_saved}" ]]; do
  printf '%s waiting for training, caches, union OOF, and save-only priority\n' "$(date -Ins)"
  sleep 30
done
for required in "${selection}" "${patch_oof}" "${manifest}" \
  "${union_calibration}" "${union_staging}/STAGING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${prewarm}" "${union_calibration}" <<'PY'
import json,sys
prewarm,baseline=[json.load(open(path,encoding="utf-8")) for path in sys.argv[1:]]
if prewarm.get("cases") != 210 or prewarm.get("component_policy") != "source_separate" or prewarm.get("labels_read") is not False:
    raise SystemExit("source-separated prewarm did not pass its audit")
if baseline.get("cases") != 210 or not baseline.get("metrics"):
    raise SystemExit("multisource union OOF calibration is incomplete")
PY

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

inference_matches() {
  local work="$1" checkpoint="$2"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(sys.argv[2]).resolve())]
if not marker.is_file(): raise SystemExit(1)
checkpoint=Path(sys.argv[2]); digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
payload=json.loads(marker.read_text())
valid=(payload.get("checkpoints",[])==expected
       and payload.get("checkpoint_sha256",[])==[digest]
       and payload.get("contour_checkpoints",[])==[]
       and payload.get("contour_checkpoint_sha256",[])==[])
raise SystemExit(0 if valid else 1)
PY
}
infer_fold() {
  local gpu="$1" fold="$2" work checkpoint
  work="$(fold_work "${fold}")"
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; return 2; }
  inference_matches "${work}" "${checkpoint}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${checkpoint}" --batch-size 8 --workers 2 \
    > "${output}/fold${fold}.infer.log" 2>&1
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_sequence 0 0 2 4 & infer0=$!
  infer_sequence 1 1 3 & infer1=$!
  status=0; wait "${infer0}" || status=$?; wait "${infer1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  infer_sequence 0 0 1 2 3 4
else
  infer_sequence 1 0 1 2 3 4
fi
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    > "${output}/fold${fold}.audit.log" 2>&1
done

classification_values=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
values=json.load(open(sys.argv[1],encoding="utf-8"))["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected exactly three thresholds")
print(" ".join(f"{float(value):.2f}" for value in values))
PY
)
segmentation=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
print(f'{float(json.load(open(sys.argv[1]))["segmentation"]["threshold"]):.2f}')
PY
)
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"

materialize_fold() {
  local fold="$1" classification="$2" mode="$3" tag="$4" work prediction
  work="$(fold_work "${fold}")"; prediction="${output}/${tag}/fold${fold}"
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${union_staging}/fold${fold}/proposals" --empty-base \
    --output "${prediction}" --classification-threshold "${classification}" \
    --segmentation-threshold "${segmentation}" --mode "${mode}" \
    --min-seed-coverage 0 --min-refined-overlap 0 \
    --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
    > "${output}/${tag}.fold${fold}.materialize.log" 2>&1
}
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
    tag="srcsep_c${classification/./p}_s${segmentation/./p}_${mode}"
    pids=(); for fold in 0 1 2 3 4; do materialize_fold "${fold}" "${classification}" "${mode}" "${tag}" & pids+=("$!"); done
    status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || exit "${status}"
    combined_dir="${output}/${tag}/combined"
    "${python_bin}" - "${output}/${tag}" "${combined_dir}" "${union_staging}/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); combined.mkdir(parents=True,exist_ok=True); seen={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in seen: raise SystemExit(f"duplicate: {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source.resolve(strict=True))
        seen[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=210 or set(seen)!=expected: raise SystemExit("OOF coverage mismatch")
PY
    metrics="${output}/${tag}.metrics.csv"; score_log="${output}/${tag}.score.log"
    "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
      --pred "${combined_dir}" --gt "${union_staging}/all/labels" \
      --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
      "${segmentation}" "${mode}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${output}/summary.tsv"
  done
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr; } > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 --seed 20260823 \
  > "${output}/summarize.log" 2>&1

anchor_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${union_calibration}")
candidate_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${calibration}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name multisource_union_oof --candidate-name source_separate_oof \
  --output "${robust}" --bootstrap 20000 --seed 20260823 \
  --min-dsc-gain 0.005 --min-nsd-gain 0.005 --min-win-rate 0.52 \
  > "${output}/paired_vs_union.log" 2>&1

"${python_bin}" - "${calibration}" "${robust}" "${prewarm}" "${complete}" <<'PY'
import datetime,json,sys
from pathlib import Path
calibration_path,robust_path,prewarm_path,output=map(Path,sys.argv[1:])
calibration=json.loads(calibration_path.read_text(encoding="utf-8"))
robust=json.loads(robust_path.read_text(encoding="utf-8"))
prewarm=json.loads(prewarm_path.read_text(encoding="utf-8"))
payload={
 "schema_version":1,
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"exact-210 case-held-out grouped-fold transfer OOF; no public or hidden data",
 "component_policy":"source_separate",
 "cases":210,
 "candidates":prewarm["candidates"],
 "Lesion_DSC":calibration["Lesion_DSC"],
 "Lesion_NSD":calibration["Lesion_NSD"],
 "classification_threshold":calibration["classification_threshold"],
 "segmentation_threshold":calibration["segmentation_threshold"],
 "deployment_mode":calibration.get("deployment_mode","refined"),
 "calibration":str(calibration_path.resolve()),
 "paired_comparison":str(robust_path.resolve()),
 "decision":"QUALIFY_FOR_FIXED_PUBLIC" if robust.get("decision")=="PROMOTE" else "KEEP_UNION",
 "public_read":False,
 "hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
cat "${output}/summary_by_dsc.tsv"
