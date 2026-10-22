#!/usr/bin/env bash
set -euo pipefail

# CPU-only final source-domain audit after exact-707 patch preparation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
patches="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/patches_bbox24_ss_min5_v4_cell48"
output="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/SOURCE_DOMAIN_AUDIT.json"

exec 9>"${root}/runs/run_candidate_refiner_v4_source_domain_audit_20260823.lock"
flock -n 9 || { echo "exact-707 source-domain audit already active" >&2; exit 1; }
[[ -s "${output}" ]] && exit 0
while [[ ! -s "${patches}/COMPLETE.txt" || ! -s "${patches}/AUDIT.json" ]]; do
  printf '%s waiting for exact-707 patch completion and audit\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" -u "${root}/audit_candidate_refiner_source_domains.py" \
  --index "${patches}/index.csv" --output "${output}" \
  --sampling-temperature 0.5 --sample-per-group 256 \
  --positive-weight-cap 30 --seed 20260823
