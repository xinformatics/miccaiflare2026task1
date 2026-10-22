#!/usr/bin/env bash
set -euo pipefail

# Public-only test of an independent prompted lesion segmenter. The prompt bank
# was fixed before labels were scored. This script cannot generate hidden masks
# or a submission archive.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
proposals="${project}/external/BiomedParse/public_sweeps/expanded_prompt_guided_20260821/cancer_mass_e050_nifti"
base="${project}/external/GF-Screen/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
output="${root}/public_sweeps/cancer_mass_prompt_refine_20260821"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "cancer-mass Lite-ENSAM audit is already active" >&2
  exit 1
fi

[[ $(find "${proposals}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

MPLCONFIGDIR="${output}/mpl" CUDA_VISIBLE_DEVICES=0 \
  "${python_bin}" -u "${root}/refine_gf_components.py" \
  --images "${images}" --proposals "${proposals}" \
  --output-root "${output}/refined_min500" --device cuda:0 \
  --threads 4 --min-proposal-mm3 500 --continue-prediction \
  > "${output}/refine.log" 2>&1

[[ $(find "${output}/refined_min500/lite" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${project}/external/GF-Screen/screen_disconnected_guided_additions.py" screen \
  --base "${base}" --candidate "${output}/refined_min500/lite" \
  --support "${proposals}" --gt "${labels}" \
  --output "${output}/screen" --workers 16 \
  > "${output}/screen.log" 2>&1

head -4 "${output}/screen/summary.tsv"
