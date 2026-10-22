#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
organ_complete="${root}/../BiomedParse/public_sweeps/organ_specific_rescue_20260821/PUBLIC_ONLY_COMPLETE.txt"
queue_lock="${root}/runs/glance_teacher_safe_queue_20260821.lock"
started_marker="${root}/runs/GLANCE_TEACHER_SAFE_STARTED_20260821.txt"

exec 9>"${queue_lock}"
if ! flock -n 9; then
  echo "teacher-safe Glance queue is already active" >&2
  exit 1
fi
while [[ ! -s "${organ_complete}" ]]; do
  sleep 30
done
date --iso-8601=seconds > "${started_marker}"
bash "${root}/run_dual_glance_teacher_safe_20260821.sh"
