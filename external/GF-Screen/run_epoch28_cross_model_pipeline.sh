#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
predecessor="public_sweeps/cross_model_component_fusion_distance/robust_selection.json"
screen="analysis/cross_model_component_fusion_epoch28_screen.json"
root="public_sweeps/cross_model_component_fusion_epoch28"
public_support="../../predictions/flare25_finetune_by_epoch/epoch_0028/predictions"
hidden_support="../../predictions/flare25_finetune_hidden_epoch_0028"
gf_public="predictions/gf_a055_bbox_pu_best_ep280/public_raw"
gf_hidden="predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
hidden_images="../../validation/Validation-Hidden-Images"

while [[ ! -s "${predecessor}" ]]; do
  sleep 15
done
mkdir -p "${root}" ../../submissions

"${python_bin}" -u screen_cross_model_component_fusion.py \
  --gf "${gf_public}" --support "${public_support}" \
  --labels ../../validation/Validation-Public-Labels \
  --distances-mm 0,2,5,10 --rescue-min-mm3 inf,1000,3000,10000 \
  --output "${screen}" > "${root}/screen.log" 2>&1

"${python_bin}" select_cross_model_configs.py \
  --input "${screen}" --output "${root}/selected_screen_configs.tsv" \
  --top 3 --max-zero 7 --require-finite-rescue > "${root}/selection.log"

printf 'distance_mm\trescue_min_mm3\tscreen_DSC\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
tail -n +2 "${root}/selected_screen_configs.tsv" | \
while IFS=$'\t' read -r distance rescue screen_dsc win_rate zeros; do
  distance_tag="${distance/./p}"
  rescue_tag="${rescue/./p}"
  candidate="${root}/d${distance_tag}_r${rescue_tag}"
  mkdir -p "${candidate}"
  "${python_bin}" -u materialize_cross_model_component_fusion.py \
    --gf "${gf_public}" --support "${public_support}" \
    --distance-mm "${distance}" --rescue-min-mm3 "${rescue}" \
    --output "${candidate}" > "${candidate}.materialize.log" 2>&1
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${candidate}" --gt ../../validation/Validation-Public-Labels \
    --csv "${candidate}.metrics.csv" --workers 4 > "${candidate}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${candidate}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${candidate}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${distance}" "${rescue}" "${screen_dsc}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r distance rescue screen_dsc dsc nsd combined <<< "${best_line}"
distance_tag="${distance/./p}"
rescue_tag="${rescue/./p}"
candidate="${root}/d${distance_tag}_r${rescue_tag}"
candidate_tag="cross_nnu28_d${distance_tag}_r${rescue_tag}"
"${python_bin}" select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${candidate}.metrics.csv" \
  --anchor-name gf_champion --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Epoch-28 cross-model candidate did not clear the robustness gate."
  exit 0
fi

hidden_output="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_output}"
"${python_bin}" -u materialize_cross_model_component_fusion.py \
  --gf "${gf_hidden}" --support "${hidden_support}" \
  --distance-mm "${distance}" --rescue-min-mm3 "${rescue}" \
  --output "${hidden_output}" > "${hidden_output}.materialize.log" 2>&1
count=$(find "${hidden_output}" -maxdepth 1 -name '*.nii.gz' | wc -l)
if [[ "${count}" -ne 100 ]]; then
  echo "Expected 100 hidden predictions, found ${count}" >&2
  exit 1
fi
submission="../../submissions/gf_bbox_ep280_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_output}" \
  --output "${submission}" --expected-count 100 > "${hidden_output}.package.log" 2>&1
unzip -t "${submission}" > "${hidden_output}.zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_output}.selected_public_configuration.tsv"
echo "Matched epoch-28 cross-model submission ready: ${submission}"
