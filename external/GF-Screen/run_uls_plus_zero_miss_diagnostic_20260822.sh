#!/usr/bin/env bash
set -euo pipefail

# Public-label diagnostic only. Four champion zero-Dice cases are held out of
# all hidden inference and submission materialization. The official ULS+ model
# receives only Rich16 Focus-0.20 component clicks during preparation/inference.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
proposals="${root}/public_sweeps/gf_rich16_comp281_low_thresholds_20260821/inference/lesion_0.20"
output="${root}/analysis/uls_plus_zero_miss_diagnostic_20260822"
work="${output}/work"
adapter="${root}/run_uls_plus_component_refiner.py"

mkdir -p "${output}" "${work}"
exec 9>"${output}.lock"
flock -n 9 || { echo "ULS+ zero-miss diagnostic already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_DIAGNOSTIC_COMPLETE.txt" ]] && exit 0

export MPLCONFIGDIR="${work}/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(
  --images "${images}"
  --proposals "${proposals}"
  --work-dir "${work}"
  --output "${output}/placeholder"
  --min-seed-mm3 20
  --max-seed-mm3 200000
  --max-components-per-case 128
  --preprocessing-workers 2
  --export-workers 2
)

if [[ ! -s "${work}/manifest.json" ]]; then
  "${python_bin}" -u "${adapter}" --stage prepare "${common[@]}" \
    --case-id FLARE23Ts_0021 \
    --case-id FLARE23Ts_0029 \
    --case-id FLARE23Ts_0039 \
    --case-id FLARE23Ts_0079 \
    > "${output}/prepare.log" 2>&1
fi
manifest_cases=$("${python_bin}" -c \
  'import json,sys; print(len(set(x["case"] for x in json.load(open(sys.argv[1])))))' \
  "${work}/manifest.json")
[[ "${manifest_cases}" -eq 4 ]] || { echo "manifest has ${manifest_cases}/4 cases" >&2; exit 2; }

while :; do
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used1}"
  sleep 30
done

if [[ ! -s "${work}/INFERENCE_COMPLETE.txt" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${adapter}" --stage infer "${common[@]}" \
    --device cuda --disable-mirroring > "${output}/inference.log" 2>&1
  predicted=$(find "${work}/predicted_crops" -maxdepth 1 -name '*.nii.gz' | wc -l)
  expected=$("${python_bin}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${work}/manifest.json")
  [[ "${predicted}" -eq "${expected}" ]] || {
    echo "ULS+ predictions incomplete: ${predicted}/${expected}" >&2
    exit 2
  }
  date -Ins > "${work}/INFERENCE_COMPLETE.txt"
fi

"${python_bin}" -u "${root}/analyze_uls_zero_miss_diagnostic.py" \
  --work "${work}" --labels "${labels}" --output "${output}/report.json" \
  > "${output}/analyze.log" 2>&1
date -Ins > "${output}/PUBLIC_DIAGNOSTIC_COMPLETE.txt"
cat "${output}/analyze.log"
