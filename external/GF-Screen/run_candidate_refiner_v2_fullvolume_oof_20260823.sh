#!/usr/bin/env bash
set -euo pipefail

# Leakage-free end-to-end OOF validation. Every case is inferred only by its
# held-out grouped-fold checkpoint. Public and hidden data are never read.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v2_training_20260822"
training_complete="${train}/TRAINING_COMPLETE.txt"
selection="${train}/checkpoint_selection.json"
oof="${train}/five_fold_oof_summary.json"
index="${train}/patches_bbox24_min5_v5/index.csv"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
proposals="${train}/epoch105_abd_g090_l050"
output="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823"
staging="${output}/staging"
calibration="${output}/deployment_calibration.json"
complete="${output}/FULL_VOLUME_OOF_COMPLETE.txt"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v2_fullvolume_oof_20260823.lock"
flock -n 9 || { echo "full-volume OOF controller already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${training_complete}" ]]; do
  printf '%s waiting for deterministic five-fold training\n' "$(date -Ins)"
  sleep 30
done
for required in "${selection}" "${oof}" "${index}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
  --index "${index}" --manifest "${manifest}" --proposals "${proposals}" \
  --output "${staging}" --folds 5 --seed 20260822 \
  > "${output}/stage.log" 2>&1
[[ -s "${staging}/STAGING_COMPLETE.json" ]]

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
prepare_fold() {
  local fold="$1" fold_stage="${staging}/fold${fold}" work
  work="$(fold_work "${fold}")"
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${fold_stage}/images" \
      --proposal "epoch105=${fold_stage}/proposals" \
      --base "${fold_stage}/proposals" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 \
      --bbox-margin-mm 24 --tile-stride-mm 72 \
      > "${output}/fold${fold}.prepare.log" 2>&1
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    > "${output}/fold${fold}.audit.log" 2>&1
}
prepare_sequence() { local fold; for fold in "$@"; do prepare_fold "${fold}"; done; }
prepare_sequence 0 2 4 & prep0=$!
prepare_sequence 1 3 & prep1=$!
status=0; wait "${prep0}" || status=$?; wait "${prep1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
free_gpus=()
while [[ ${#free_gpus[@]} -eq 0 ]]; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free_gpus=()
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free_gpus+=(0)
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free_gpus+=(1)
  if [[ ${#free_gpus[@]} -eq 0 ]]; then
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  fi
done

inference_matches() {
  local work="$1" checkpoint="$2"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" <<'PY'
import json,sys
import hashlib
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(sys.argv[2]).resolve())]
if not marker.is_file(): raise SystemExit(1)
checkpoint=Path(sys.argv[2]); digest=hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
payload=json.loads(marker.read_text(encoding="utf-8"))
valid=(payload.get("checkpoints")==expected
       and payload.get("checkpoint_sha256")==[digest.hexdigest()]
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
infer_sequence() {
  local gpu="$1"; shift
  local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done
}
status=0
if [[ ${#free_gpus[@]} -ge 2 ]]; then
  infer_sequence "${free_gpus[0]}" 0 2 4 & infer0=$!
  infer_sequence "${free_gpus[1]}" 1 3 & infer1=$!
  wait "${infer0}" || status=$?; wait "${infer1}" || status=$?
else
  # Reuse matching prewarmed fold caches and serialize only missing inference
  # on the currently free device. This avoids waiting for an unrelated GPU job.
  infer_sequence "${free_gpus[0]}" 0 1 2 3 4 || status=$?
fi
[[ "${status}" -eq 0 ]] || exit "${status}"

read -r classification_center classification_values < <("${python_bin}" - "${oof}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); values=p["classification"]["deployment_thresholds"]
if len(values) != 3 or len(set(map(float,values))) != 3: raise SystemExit("expected exactly three frozen deployment thresholds")
print(f'{float(p["classification"]["threshold"]):.2f}', ",".join(f"{value:.2f}" for value in values))
PY
)
classification_values="${classification_values//,/ }"
read -r segmentation_center segmentation_values < <(PYTHONPATH="${root}" "${python_bin}" - "${oof}" <<'PY'
import json,sys
from summarize_candidate_refiner_oof import deployment_segmentation_thresholds
p=json.load(open(sys.argv[1],encoding="utf-8")); center=float(p["segmentation"]["threshold"])
values=deployment_segmentation_thresholds(center)
print(f"{center:.2f}", ",".join(f"{value:.2f}" for value in values))
PY
)
segmentation_values="${segmentation_values//,/ }"

materialize_grid() {
  local stage="$1"; shift
  local specs=("$@") variant_args=() spec fold status=0
  local pids=()
  for spec in "${specs[@]}"; do variant_args+=(--variant "${spec}"); done
  for fold in 0 1 2 3 4; do
    "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
      --work "$(fold_work "${fold}")" \
      --base "${staging}/fold${fold}/proposals" --output-root "${output}" \
      --fold "${fold}" "${variant_args[@]}" \
      > "${output}/${stage}.fold${fold}.grid.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || return "${status}"
}
score_variant() {
  local classification="$1" segmentation_value="$2" mode="$3" tag="$4" summary="$5"
  local combined_dir metrics score_log dsc nsd combined
  combined_dir="${output}/${tag}/combined"
  "${python_bin}" - "${output}/${tag}" "${combined_dir}" "${staging}/all/labels" <<'PY'
import os,sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); combined.mkdir(parents=True,exist_ok=True)
observed={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in observed: raise SystemExit(f"duplicate prediction: {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source.resolve(strict=True))
        observed[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=210 or set(observed)!=expected:
    raise SystemExit(f"OOF coverage mismatch expected={len(expected)} observed={len(observed)} missing={sorted(expected-set(observed))[:3]}")
PY
  metrics="${output}/${tag}.metrics.csv"
  score_log="${output}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined_dir}" --gt "${staging}/all/labels" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation_value}" "${mode}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" >> "${summary}"
}

# The patch-level optimum is at the top of its evaluated grid. Audit two
# higher, predeclared contour thresholds on exact held-out full volumes before
# freezing the final segmentation threshold. This adds no public-label search.
segmentation_sweep="${output}/segmentation_threshold_sweep.tsv"
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${segmentation_sweep}"
segmentation_specs=()
for segmentation_value in ${segmentation_values}; do
  tag="segcal_c${classification_center/./p}_s${segmentation_value/./p}_refined"
  segmentation_specs+=("${tag},${classification_center},${segmentation_value},refined")
done
materialize_grid segmentation_calibration "${segmentation_specs[@]}"
for segmentation_value in ${segmentation_values}; do
  tag="segcal_c${classification_center/./p}_s${segmentation_value/./p}_refined"
  score_variant "${classification_center}" "${segmentation_value}" refined "${tag}" "${segmentation_sweep}"
done
segmentation=$("${python_bin}" -u "${root}/select_segmentation_threshold_sweep.py" \
  --summary "${segmentation_sweep}" --patch-oof-center "${segmentation_center}" \
  --output "${output}/SEGMENTATION_THRESHOLD_SELECTION.json")

printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
deployment_specs=()
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
    tag="stand_c${classification/./p}_s${segmentation/./p}_${mode}"
    deployment_specs+=("${tag},${classification},${segmentation},${mode}")
  done
done
materialize_grid deployment "${deployment_specs[@]}"
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
    tag="stand_c${classification/./p}_s${segmentation/./p}_${mode}"
    score_variant "${classification}" "${segmentation}" "${mode}" "${tag}" "${output}/summary.tsv"
  done
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" \
  --checkpoint-selection "${selection}" --manifest "${manifest}" \
  --output "${calibration}" --bootstrap-samples 5000 --seed 20260823 \
  > "${output}/summarize.log" 2>&1
date -Ins > "${complete}"
cat "${output}/summary_by_dsc.tsv"
