#!/usr/bin/env bash
set -euo pipefail

# Recall-oriented complement to component confirmation. Preserve the filtered
# GF champion and add only selected connected components from the matched
# nnU-Net epoch-28 public/hidden predictions.
wait_pid="${WAIT_PID:-853817}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
gf_public="predictions/gf_a055_bbox_pu_best_ep280/public_mm1000"
gf_hidden="predictions/gf_a055_bbox_pu_best_ep280/hidden_mm1000"
support_public="../../predictions/flare25_finetune_by_epoch/epoch_0028/predictions"
support_hidden="../../predictions/flare25_finetune_hidden_epoch_0028"
public_labels="../../validation/Validation-Public-Labels"
hidden_images="../../validation/Validation-Hidden-Images"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
screen="analysis/cross_model_component_addition_epoch28_screen.json"
root="public_sweeps/cross_model_component_addition_epoch28"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
mkdir -p "${root}" ../../submissions
if [[ ! -s "${screen}" ]]; then
  "${python_bin}" -u screen_cross_model_component_addition.py \
    --base "${gf_public}" --support "${support_public}" --labels "${public_labels}" \
    --min-mm3 0,100,500,1000,3000,10000 --modes disjoint,all \
    --output "${screen}" > "${root}/screen.log" 2>&1
fi
screen_decision=$("${python_bin}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print("CONTINUE" if d["summary"][0]["mean_dice"] > d["baseline_mean_dice"] else "REJECT")' \
  "${screen}")
if [[ "${screen_decision}" != "CONTINUE" ]]; then
  "${python_bin}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); b=d["baseline_mean_dice"]; r=d["summary"][0]; print("baseline_DSC={:.8f}\tbest_mode={}\tbest_min_mm3={:g}\tbest_DSC={:.8f}".format(b,r["mode"],r["min_mm3"],r["mean_dice"]))' \
    "${screen}" > "${root}/screen_not_promoted.tsv"
  echo "Recall-oriented cross-model addition is below baseline at screen stage."
  exit 0
fi
"${python_bin}" select_cross_addition_configs.py \
  --input "${screen}" --output "${root}/selected_screen_configs.tsv" \
  --top 3 --max-zero 4 --max-catastrophic-rate 0.10 > "${root}/selection.log"

printf 'mode\tmin_mm3\tscreen_DSC\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
tail -n +2 "${root}/selected_screen_configs.tsv" | \
while IFS=$'\t' read -r mode min_mm3 screen_dsc win_rate zeros catastrophic; do
  mm_tag="${min_mm3/./p}"
  candidate="${root}/${mode}_mm${mm_tag}"
  mkdir -p "${candidate}"
  "${python_bin}" -u materialize_cross_model_component_addition.py \
    --base "${gf_public}" --support "${support_public}" --mode "${mode}" \
    --min-mm3 "${min_mm3}" --output "${candidate}" > "${candidate}.materialize.log" 2>&1
  "${python_bin}" -u evaluate_public_parallel.py \
    --pred "${candidate}" --gt "${public_labels}" \
    --csv "${candidate}.metrics.csv" --workers 4 > "${candidate}.score.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${candidate}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${candidate}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${min_mm3}" "${screen_dsc}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r mode min_mm3 screen_dsc dsc nsd combined <<< "${best_line}"
mm_tag="${min_mm3/./p}"
candidate="${root}/${mode}_mm${mm_tag}"
candidate_tag="cross_add_nnu28_${mode}_mm${mm_tag}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${candidate}.metrics.csv" \
  --anchor-name gf_champion --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Recall-oriented cross-model addition did not clear the robustness gate."
  exit 0
fi

hidden_output="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_output}"
"${python_bin}" -u materialize_cross_model_component_addition.py \
  --base "${gf_hidden}" --support "${support_hidden}" --mode "${mode}" \
  --min-mm3 "${min_mm3}" --output "${hidden_output}" > "${hidden_output}.materialize.log" 2>&1
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
echo "Recall-oriented cross-model submission ready: ${submission}"
