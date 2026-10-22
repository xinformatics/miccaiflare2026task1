#!/usr/bin/env bash
set -euo pipefail

# Controlled zero-shot audit: use the frozen GF champion components only as
# 3D box prompts, then score the refiner alone and conservative anchor fusions.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
gpu="${GPU:-0}"
smoke_only="${SMOKE_ONLY:-1}"
model_root="${MODEL_ROOT:-${script_dir}/LesionLocatorCheckpoint}"
images="${PUBLIC_IMAGES:-${workspace}/validation/Validation-Public-Images}"
labels="${PUBLIC_LABELS:-${workspace}/validation/Validation-Public-Labels}"
anchor="${ANCHOR:-${workspace}/external/GF-Screen/public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000}"
output="${OUTPUT_ROOT:-${script_dir}/public_audit/champion_mm1000_fold0_notta}"

# The shared conda environment inherits Intel MKL, while PyTorch and
# LesionLocator's spawned preprocessors load GNU OpenMP. Make the choice
# explicit so multiprocessing is reproducible instead of dying at import.
export MKL_THREADING_LAYER=GNU

checkpoint="${model_root}/LesionLocatorSeg/bbox_optimized/fold_0/checkpoint_final.pth"
[[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
[[ "${smoke_only}" == "0" || "${smoke_only}" == "1" ]] || {
  echo "SMOKE_ONLY must be 0 or 1" >&2
  exit 1
}
if [[ -s "${output}/summary_by_combined.tsv" \
   && -s "${output}/inference_seconds.txt" ]]; then
  echo "LesionLocator audit already complete: ${output}"
  cat "${output}/summary_by_combined.tsv"
  exit 0
fi

mkdir -p "${output}/inputs" "${output}/prompts" "${output}/instances" \
  "${output}/merged" "${output}/scored"
if [[ "${smoke_only}" == "1" ]]; then
  cases=(0001 0005 0021 0073 0099)
  for case_id in "${cases[@]}"; do
    ln -sfn "${images}/FLARE23Ts_${case_id}_0000.nii.gz" \
      "${output}/inputs/FLARE23Ts_${case_id}_0000.nii.gz"
    ln -sfn "${anchor}/FLARE23Ts_${case_id}.nii.gz" \
      "${output}/prompts/FLARE23Ts_${case_id}.nii.gz"
  done
  expected=5
else
  for source in "${images}"/*.nii.gz; do
    ln -sfn "${source}" "${output}/inputs/$(basename "${source}")"
  done
  for source in "${anchor}"/*.nii.gz; do
    ln -sfn "${source}" "${output}/prompts/$(basename "${source}")"
  done
  expected=50
fi

start_seconds=$(date +%s)
CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${script_dir}" "${python_bin}" -u \
  -c 'from lesionlocator.inference.lesionlocator_segment import predict_seg_from_prompt; predict_seg_from_prompt()' \
  -i "${output}/inputs" -p "${output}/prompts" -t box \
  -o "${output}/instances" -m "${model_root}" -f 0 \
  --disable_tta --disable_progress_bar -npp 2 -nps 2 \
  > "${output}/inference.log" 2>&1
end_seconds=$(date +%s)
printf '%s\n' "$((end_seconds - start_seconds))" > "${output}/inference_seconds.txt"

printf 'mode\tmin_mm3\tDSC\tNSD\tcombined\n' > "${output}/summary.tsv"
"${python_bin}" "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
  --pred "${output}/prompts" --gt "${labels}" \
  --csv "${output}/scored/anchor_mm1000.metrics.csv" --workers 4 \
  > "${output}/scored/anchor_mm1000.score.log"
anchor_dsc=$(awk '/DSC  mean/ {print $3}' "${output}/scored/anchor_mm1000.score.log")
anchor_nsd=$(awk '/NSD  mean/ {print $3}' "${output}/scored/anchor_mm1000.score.log")
anchor_combined=$(awk -v d="${anchor_dsc}" -v n="${anchor_nsd}" \
  'BEGIN {printf "%.6f", (d+n)/2}')
printf 'anchor\t1000\t%s\t%s\t%s\n' \
  "${anchor_dsc}" "${anchor_nsd}" "${anchor_combined}" >> "${output}/summary.tsv"
for mode in refiner union intersection; do
  merged="${output}/merged/${mode}"
  mode_args=()
  if [[ "${mode}" != "refiner" ]]; then
    mode_args+=(--anchor "${anchor}")
  fi
  "${python_bin}" "${script_dir}/merge_instance_predictions.py" \
    --references "${output}/inputs" --instances "${output}/instances" \
    --output "${merged}" --mode "${mode}" --expected-count "${expected}" \
    "${mode_args[@]}" > "${output}/merge_${mode}.log"
  for min_mm3 in 0 500 1000; do
    candidate="${merged}"
    score_stem="${output}/scored/${mode}_mm${min_mm3}"
    if [[ "${min_mm3}" -gt 0 ]]; then
      candidate="${score_stem}"
      mkdir -p "${candidate}"
      "${python_bin}" "${workspace}/external/GF-Screen/fuse_and_filter_masks.py" \
        --pred-a "${merged}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${candidate}" > "${score_stem}.filter.log"
    fi
    "${python_bin}" "${workspace}/external/GF-Screen/evaluate_public_parallel.py" \
      --pred "${candidate}" --gt "${labels}" --csv "${score_stem}.metrics.csv" \
      --workers 4 > "${score_stem}.score.log"
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${mode}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" >> "${output}/summary.tsv"
  done
done

{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr; } \
  > "${output}/summary_by_combined.tsv"
cat "${output}/summary_by_combined.tsv"
