#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
started_marker="${root}/runs/GLANCE_TEACHER_SAFE_STARTED_20260821.txt"

exec 9>"${root}/runs/glance_teacher_safe_eval_queue_20260821.lock"
if ! flock -n 9; then
  echo "teacher-safe Glance evaluation queue is already active" >&2
  exit 1
fi
while [[ ! -s "${started_marker}" ]]; do
  sleep 30
done
exec 8>"${root}/runs/dual_glance_teacher_safe_20260821.lock"
flock 8
bash "${root}/evaluate_dual_glance_teacher_safe_public_20260821.sh"
