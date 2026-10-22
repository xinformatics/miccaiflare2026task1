#!/usr/bin/env bash
set -euo pipefail

# CPU/public-only nested grouped selector over ULS+ inference-visible features.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/public_sweeps/uls_plus_rich20_refiner_20260822/work"
uls_marker="${work}/INFERENCE_COMPLETE.txt"
components="${root}/analysis/component_classifier_cv_20260822/components_oof.csv"
component_marker="${root}/analysis/component_classifier_cv_20260822/COMPLETE.txt"
output="${root}/analysis/uls_feature_classifier_cv_20260822"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "ULS+ feature classifier already active" >&2; exit 1; }
[[ -s "${output}/COMPLETE.txt" ]] && exit 0
while [[ ! -s "${uls_marker}" || ! -s "${component_marker}" ]]; do
  printf '%s waiting for ULS+ inference and component features\n' "$(date -Ins)"
  sleep 30
done

if [[ ! -s "${output}/report.json" ]]; then
  cd "${root}"
  "${python_bin}" -u analyze_uls_feature_classifier_cv.py \
    --images "${project}/validation/Validation-Public-Images" \
    --labels "${project}/validation/Validation-Public-Labels" \
    --base "${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25" \
    --work "${work}" --component-csv "${components}" \
    --support seed40="${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.40" \
    --support seed55="${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.55" \
    --support richfull="${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281/inference/lesion_0.60" \
    --support whole_prompt="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti" \
    --support tumor_prompt="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti" \
    --support nnunet="${root}/public_sweeps/complementary_guided_union_20260821/nnunet_new2k_s25_sd10" \
    --output-dir "${output}" --seed 20260822 > "${output}/run.log" 2>&1
fi

[[ $(find "${output}/predictions_oof" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${output}/predictions_oof" \
  --gt "${project}/validation/Validation-Public-Labels" \
  --csv "${output}/metrics.csv" --workers 2 > "${output}/score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${output}/metrics.csv" \
  --anchor-name champion_0p7006 --candidate-name uls_feature_classifier_grouped_oof \
  --output "${output}/robust.json" > "${output}/robust.log"
date -Ins > "${output}/COMPLETE.txt"
