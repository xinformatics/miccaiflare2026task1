#!/usr/bin/env bash
set -euo pipefail

# One boundary extension because the predeclared 0.5--5.0 cm3 standalone
# filter grid improved monotonically through its upper endpoint.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
source="${root}/public_sweeps/expanded_prompt_guided_20260821/malignant_metastasis_e050_nifti"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/standalone_malignant_filter_extension_20260821"
thresholds=(7500 10000 15000 20000 30000)

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "standalone malignant filter extension is already active" >&2
  exit 1
fi
[[ $(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
args=()
for threshold in "${thresholds[@]}"; do
  destination="${output}/mm${threshold}"
  mkdir -p "${destination}"
  "${python_bin}" -u "${gf}/fuse_and_filter_masks.py" \
    --pred-a "${source}" --fusion a --min-mm3 "${threshold}" --workers 4 \
    --output "${destination}" > "${output}/mm${threshold}.filter.log" 2>&1
  [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  args+=(--candidate "mm${threshold}=${destination}")
done
"${python_bin}" -u "${gf}/evaluate_public_grid.py" \
  --gt "${labels}" --output-dir "${output}/metrics" --workers 8 \
  "${args[@]}" > "${output}/score.log" 2>&1
printf 'min_mm3\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
for threshold in "${thresholds[@]}"; do
  metrics="${output}/metrics/mm${threshold}.metrics.csv"
  dsc=$(awk -F, 'NR>1 {s+=$2;n++} END {printf "%.6f",s/n}' "${metrics}")
  nsd=$(awk -F, 'NR>1 {s+=$3;n++} END {printf "%.6f",s/n}' "${metrics}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "${threshold}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k2,2gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
