#!/usr/bin/env bash
set -euo pipefail

# Preserve the older exact-210 source-separated comparison without allowing it
# to race the critical 707-case proposal producer for the first free GPU.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
epoch105_complete="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050/PROPOSALS_COMPLETE.json"
while [[ ! -s "${epoch105_complete}" ]]; do
  printf '%s waiting for critical exact-707 epoch105 proposals\n' "$(date -Ins)"
  sleep 30
done
exec bash "${root}/run_candidate_refiner_source_separate_transfer_oof_20260823.sh"
