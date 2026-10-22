#!/usr/bin/env bash
set -euo pipefail

# The multisource controller was already running when its materialization stage
# was upgraded to the byte-parity-tested shared sparse implementation. Wait for
# all expensive GPU inference caches to finish, stop only that older process
# group, and restart the controller so it loads the audited implementation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
analysis="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
controller="${root}/run_candidate_refiner_v2_multisource_after_primary_20260823.sh"
old_pgid="2818288"

mkdir -p "${analysis}"
exec 9>"${root}/runs/handoff_multisource_to_sparse_20260823.lock"
flock -n 9 || exit 0

while :; do
  complete=1
  for fold in 0 1 2 3 4; do
    [[ -s "${analysis}/fold${fold}/work_bbox24/INFERENCE_COMPLETE.json" ]] || complete=0
  done
  [[ "${complete}" -eq 1 ]] && break
  sleep 2
done

if [[ -r "/proc/${old_pgid}/cmdline" ]] \
    && tr '\0' ' ' < "/proc/${old_pgid}/cmdline" \
      | grep -Fq 'run_candidate_refiner_v2_multisource_after_primary_20260823.sh'; then
  kill -TERM -- "-${old_pgid}"
  for _ in $(seq 1 30); do
    kill -0 "${old_pgid}" 2>/dev/null || break
    sleep 1
  done
fi

while ! flock -n "${root}/runs/run_candidate_refiner_v2_multisource_after_primary_20260823.lock" true 2>/dev/null; do
  sleep 1
done
sleep 2
cd "${project}"
exec bash "${controller}" >> "${analysis}/sparse_restart.log" 2>&1
