#!/usr/bin/env bash
set -euo pipefail

# Exact organizer-metric comparison of cached BiomedParse-only masks. No GF
# mask, checkpoint, component verifier, hidden data, or submission packaging is
# used by this audit.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
labels="${project}/validation/Validation-Public-Labels"
old="${root}/public_sweeps/automatic_text_proposals_20260820"
combined="${root}/public_sweeps/combined_prompt_confirmation_20260820/supports"
expanded="${root}/public_sweeps/expanded_prompt_guided_20260821"
output="${root}/public_sweeps/standalone_prompt_banks_20260821"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "standalone BiomedParse audit is already active" >&2
  exit 1
fi

declare -A candidates=(
  [prompt_or_e050]="${combined}/union_e050"
  [prompt_and_e050]="${combined}/intersection_e050"
  [cancer_mass_e050]="${expanded}/cancer_mass_e050_nifti"
  [malignant_metastasis_e050]="${expanded}/malignant_metastasis_e050_nifti"
)
args=()
for tag in prompt_or_e050 prompt_and_e050 cancer_mass_e050 malignant_metastasis_e050; do
  directory="${candidates[${tag}]}"
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  args+=(--candidate "${tag}=${directory}")
done

"${python_bin}" -u "${gf}/evaluate_public_grid.py" \
  --gt "${labels}" --output-dir "${output}/metrics" --workers 8 \
  "${args[@]}" > "${output}/score.log" 2>&1

printf 'variant\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
for tag in prompt_or_e050 prompt_and_e050 cancer_mass_e050 malignant_metastasis_e050; do
  metrics="${output}/metrics/${tag}.metrics.csv"
  dsc=$(awk -F, 'NR>1 {s+=$2;n++} END {printf "%.6f",s/n}' "${metrics}")
  nsd=$(awk -F, 'NR>1 {s+=$3;n++} END {printf "%.6f",s/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "${tag}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k2,2gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
