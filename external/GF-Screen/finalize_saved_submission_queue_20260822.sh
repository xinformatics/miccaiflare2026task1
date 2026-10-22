#!/usr/bin/env bash
set -euo pipefail

# Consolidate and integrity-check the save-only strict and backup queues.
# This controller never uploads to Codabench.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cross="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822"
upstream="${cross}/SAVED_BACKUPS_COMPLETE.txt"
output="${project}/submissions/READY_20260822.tsv"
audit="${project}/submissions/READY_20260822.audit.json"
marker="${project}/submissions/READY_20260822_COMPLETE.txt"

exec 9>"${project}/submissions/finalize_saved_submission_queue_20260822.lock"
flock -n 9 || { echo "submission finalizer already active" >&2; exit 1; }
[[ -s "${marker}" ]] && exit 0
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for saved hybrid backup queue\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" "${root}/finalize_saved_submission_queue.py" \
  --strict "${cross}/saved_submissions.tsv" \
  --backup "${cross}/saved_backup_submissions.tsv" \
  --submitted "${project}/submissions/codabench_submitted_sha256.tsv" \
  --output "${output}" --audit-output "${audit}" \
  --expected-count 100 --max-count 3 \
  > "${project}/submissions/READY_20260822.log" 2>&1
date -Ins > "${marker}"
