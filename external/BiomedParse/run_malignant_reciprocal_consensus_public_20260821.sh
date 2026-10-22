#!/usr/bin/env bash
set -euo pipefail

# Public-only reciprocal-consensus audit.  The deployed champion remains
# immutable; only completely disconnected components may be added.  Direction
# A asks GF to verify BiomedParse proposals, while direction B asks BiomedParse
# to verify raw GF proposals.  This script performs no hidden inference and
# creates no submission archive.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
base="${gf}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
raw_gf="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
malignant="${root}/public_sweeps/expanded_prompt_guided_20260821/malignant_metastasis_e050_nifti"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/malignant_reciprocal_consensus_20260821"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "malignant reciprocal consensus audit is already active" >&2
  exit 1
fi
for directory in "${base}" "${raw_gf}" "${malignant}" "${labels}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 masks: ${directory}" >&2
    exit 1
  }
done

screen_direction() {
  local tag="$1" candidate="$2" support="$3"
  "${python_bin}" -u "${gf}/screen_disconnected_guided_additions.py" screen \
    --base "${base}" --candidate "${candidate}" --support "${support}" \
    --gt "${labels}" --output "${output}/${tag}" --workers 4 \
    > "${output}/${tag}.log" 2>&1
}

screen_direction biomed_candidate_gf_support "${malignant}" "${raw_gf}" & a=$!
screen_direction gf_candidate_biomed_support "${raw_gf}" "${malignant}" & b=$!
status=0
wait "${a}" || status=$?
wait "${b}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

printf 'direction\t' > "${output}/summary.tsv"
head -1 "${output}/biomed_candidate_gf_support/summary.tsv" >> "${output}/summary.tsv"
for tag in biomed_candidate_gf_support gf_candidate_biomed_support; do
  tail -n +2 "${output}/${tag}/summary.tsv" | sed "s/^/${tag}\t/" \
    >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k11,11gr -k9,9gr
} > "${output}/summary_by_combined.tsv"
head -17 "${output}/summary_by_combined.tsv"
