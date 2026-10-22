#!/usr/bin/env bash
set -euo pipefail

# Public-only alternative-verifier audit. Preserve the full current champion
# and consider only disconnected raw-GF components supported by the fixed
# cancer/abnormal-mass prompt bank. No hidden inference or packaging occurs.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
base="${gf}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
candidate="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
support="${root}/public_sweeps/expanded_prompt_guided_20260821/cancer_mass_e050_nifti"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/cancer_mass_raw_gf_rescue_20260821"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "cancer-mass raw-GF rescue audit is already active" >&2
  exit 1
fi
for directory in "${base}" "${candidate}" "${support}" "${labels}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 masks: ${directory}" >&2
    exit 1
  }
done

"${python_bin}" -u "${gf}/screen_disconnected_guided_additions.py" screen \
  --base "${base}" --candidate "${candidate}" --support "${support}" \
  --gt "${labels}" --output "${output}/screen" --workers 16 \
  > "${output}/screen.log" 2>&1

head -4 "${output}/screen/summary.tsv"
