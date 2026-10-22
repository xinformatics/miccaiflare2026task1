#!/usr/bin/env bash
set -euo pipefail

# Public-only evaluation of the epoch-105 bbox+24-mm tiled refiner. Training
# thresholds come exclusively from grouped five-fold OOF predictions. Public
# labels are read only after the inference cache and thresholds are frozen.
# This controller cannot read hidden images, create a ZIP, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train_work="${root}/analysis/candidate_refiner_v2_training_20260822"
training_complete="${train_work}/TRAINING_COMPLETE.txt"
oof="${train_work}/five_fold_oof_summary.json"
selection="${train_work}/checkpoint_selection.json"
full_oof_root="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823"
full_oof_complete="${full_oof_root}/FULL_VOLUME_OOF_COMPLETE.txt"
deployment_calibration="${full_oof_root}/deployment_calibration.json"
public_images="${project}/validation/Validation-Public-Images"
public_labels="${project}/validation/Validation-Public-Labels"
public_proposals="${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions"
public_base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/candidate_refiner_v2_bbox24_20260822"
work="${output}/work_all_bbox24"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}" "${work}"
exec 9>"${root}/runs/run_candidate_refiner_v2_public_20260822.lock"
flock -n 9 || { echo "candidate-refiner v2 public evaluator already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${training_complete}" || ! -s "${full_oof_complete}" ]]; do
  printf '%s waiting for candidate-refiner v2 training and full-volume OOF\n' "$(date -Ins)"
  sleep 30
done

for directory in "${public_images}" "${public_labels}" "${public_proposals}" "${public_base}"; do
  count=$(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq 50 ]] || { echo "expected 50 NIfTI files, found ${count}: ${directory}" >&2; exit 2; }
done
for required in "${oof}" "${selection}" "${deployment_calibration}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
checkpoint_args=()
checkpoint_paths=()
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  checkpoint_args+=(--checkpoint "${checkpoint}")
  checkpoint_paths+=("${checkpoint}")
done

export MPLCONFIGDIR="${output}/matplotlib"
export TMPDIR="/tmp/gfref_v2_public_20260822"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"

if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${public_images}" --proposal "epoch105=${public_proposals}" \
    --base "${public_base}" --work "${work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${output}/prepare_public.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 4 \
  > "${output}/audit_public_cache.log" 2>&1

inference_matches() {
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "$@" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1])
paths=[Path(path) for path in sys.argv[2:]]
expected=[str(path.resolve()) for path in paths]
if not marker.is_file(): raise SystemExit(1)
digests=[]
for path in paths:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): value.update(block)
    digests.append(value.hexdigest())
payload=json.loads(marker.read_text(encoding="utf-8"))
valid=(payload.get("checkpoints")==expected
       and payload.get("checkpoint_sha256")==digests
       and payload.get("contour_checkpoints",[])==[]
       and payload.get("contour_checkpoint_sha256",[])==[])
raise SystemExit(0 if valid else 1)
PY
}

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPU 0: %s MiB\n' "$(date -Ins)" "${used0}"
  sleep 30
done
if ! inference_matches "${checkpoint_paths[@]}"; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${output}/infer_public.log" 2>&1
fi

classification_values=$("${python_bin}" - "${deployment_calibration}" <<'PY'
import json,sys
value=float(json.load(open(sys.argv[1], encoding="utf-8"))["classification_threshold"])
print(f"{value:.2f}")
PY
)
segmentation=$("${python_bin}" - "${deployment_calibration}" <<'PY'
import json,sys
value=float(json.load(open(sys.argv[1], encoding="utf-8"))["segmentation_threshold"])
print(f"{value:.2f}")
PY
)
deployment_mode=$("${python_bin}" - "${deployment_calibration}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("deployment_mode", "refined"))
PY
)
printf '%s\n' "${classification_values}" > "${output}/classification_thresholds.txt"
printf '%s\n' "${segmentation}" > "${output}/segmentation_threshold.txt"
printf '%s\n' "${deployment_mode}" > "${output}/deployment_mode.txt"
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"

score_variant() {
  local tag="$1" classification="$2" mode="$3" deployment="$4" prediction="$5"
  local metrics="${prediction}.metrics.csv" score_log="${prediction}.score.log"
  local robust="${prediction}.robust.json" dsc nsd combined decision
  [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${prediction}" --gt "${public_labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${mode}" "${deployment}" \
    "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}" \
    >> "${output}/summary.tsv"
}

for classification in ${classification_values}; do
  ctag=${classification/./p}
  anchor_tag="v2_anchor_${deployment_mode}_c${ctag}_s${segmentation/./p}"
  standalone_tag="v2_standalone_${deployment_mode}_c${ctag}_s${segmentation/./p}"
  "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
    --work "${work}" --base "${public_base}" --output-root "${output}" \
    --flat-output \
    --variant "${anchor_tag},${classification},${segmentation},${deployment_mode},base" \
    --variant "${standalone_tag},${classification},${segmentation},${deployment_mode},empty" \
    > "${output}/public_anchor_standalone.grid.log" 2>&1
  score_variant "${anchor_tag}" "${classification}" "${deployment_mode}" anchor \
    "${output}/${anchor_tag}"
  score_variant "${standalone_tag}" "${classification}" "${deployment_mode}" standalone \
    "${output}/${standalone_tag}"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"
