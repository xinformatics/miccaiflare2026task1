#!/usr/bin/env bash
set -euo pipefail

# Public-only diagnostic: TotalSegmentator anatomy features added to the
# case-grouped disconnected-component verifier. It never sees hidden data and
# never creates or uploads a submission.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
totalseg="/home/m348911/miniconda3/envs/radoncAI/bin/TotalSegmentator"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
anatomy="${root}/public_sweeps/totalseg_fast_public_20260822"
output="${root}/analysis/component_classifier_anatomy_cv_20260822"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

mkdir -p "${anatomy}" "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "anatomy component diagnostic already active" >&2; exit 1; }
[[ -s "${output}/COMPLETE.txt" ]] && exit 0
[[ -x "${totalseg}" && -s "${anchor_metrics}" ]]
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 50 ]]

for image in "${images}"/*_0000.nii.gz; do
  case_name=$(basename "${image}" _0000.nii.gz)
  mask="${anatomy}/${case_name}.nii.gz"
  if [[ -s "${mask}" ]] && "${python_bin}" -c 'import nibabel as n,sys; n.load(sys.argv[1]).get_fdata(caching="unchanged")' "${mask}"; then
    continue
  fi
  "${totalseg}" -i "${image}" -o "${mask}" --ml --fast \
    --device gpu:0 --nr_thr_resamp 4 --nr_thr_saving 4 --quiet \
    > "${anatomy}/${case_name}.log" 2>&1
done
[[ $(find "${anatomy}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
date -Ins > "${anatomy}/INFERENCE_COMPLETE.txt"

cd "${root}"
"${python_bin}" -u analyze_component_classifier_cv.py \
  --images "${images}" --labels "${labels}" \
  --base "${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25" \
  --candidate "${root}/public_sweeps/gf_rich16_comp281_low_thresholds_20260821/inference/lesion_0.20" \
  --support seed40="${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.40" \
  --support seed55="${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.55" \
  --support richfull="${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281/inference/lesion_0.60" \
  --support whole_prompt="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti" \
  --support tumor_prompt="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti" \
  --support nnunet="${root}/public_sweeps/complementary_guided_union_20260821/nnunet_new2k_s25_sd10" \
  --anatomy-dir "${anatomy}" --output-dir "${output}" --seed 20260822 \
  > "${output}/run.log" 2>&1
[[ $(find "${output}/predictions_oof" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${output}/predictions_oof" --gt "${labels}" \
  --csv "${output}/metrics.csv" --workers 4 > "${output}/score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${output}/metrics.csv" \
  --anchor-name champion_0p7006 --candidate-name component_anatomy_grouped_oof \
  --output "${output}/robust.json" > "${output}/robust.log"
date -Ins > "${output}/COMPLETE.txt"
cat "${output}/report.json"
