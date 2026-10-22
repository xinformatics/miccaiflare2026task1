#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run="${root}/runs/candidate_refiner_base48_fold0_20260823"
analysis="${root}/analysis/candidate_refiner_base48_pilot_20260823"

mkdir -p "${analysis}"
exec 9>"${root}/runs/watch_candidate_refiner_base48_pilot_20260823.lock"
flock -n 9 || { echo "base48 pilot watcher already active" >&2; exit 1; }
[[ -s "${analysis}/DECISION.json" ]] && exit 0
while [[ ! -s "${run}/COMPLETE.txt" ]]; do
  printf '%s waiting for base48 fold-0 pilot\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${run}/checkpoint_best.pt" ]] || { echo "missing best checkpoint" >&2; exit 2; }
"${python_bin}" -u "${root}/select_candidate_refiner_base48_pilot.py" \
  --plan "${analysis}/PLAN.json" \
  --checkpoint "${run}/checkpoint_best.pt" \
  --output "${analysis}/DECISION.json"

