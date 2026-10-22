#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${project_root}/external/FLARE25-Task1-winner"
inference_terminal="${root}/exact707_primary_audit_20260828/INFERENCE_TERMINAL.json"
output="${root}/exact707_primary_audit_20260828/positive_only_comparison"
summary="${output}/summary.json"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${output}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "FLARE25 exact-707 analyzer already active" >&2; exit 1; }
while [[ ! -s "${inference_terminal}" ]]; do
  sleep 60
done
[[ $("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${inference_terminal}") == COMPLETE ]] || {
  echo "FLARE25 exact-707 inference terminal is not COMPLETE" >&2; exit 2;
}
[[ -s "${summary}" ]] && exit 0

exec >>"${output}/analysis.log" 2>&1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${python_bin}" "${root}/analyze_exact707_expert.py" \
  --manifest "${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/manifest.json" \
  --flare "${root}/exact707_primary_audit_20260828/predictions" \
  --gf "${project_root}/external/GF-Screen/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050" \
  --policy "${root}/EXACT707_EXPERT_POLICY_20260828.json" \
  --output "${output}" \
  --workers 4
