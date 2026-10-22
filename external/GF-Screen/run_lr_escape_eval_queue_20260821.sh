#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

# Prevent duplicate waiters. The training launcher holds the second lock for
# its full lifetime; acquiring it therefore creates an exact training/eval
# boundary without polling PIDs or relying on process namespaces.
exec 9>"runs/lr_escape_eval_queue_20260821.lock"
if ! flock -n 9; then
  echo "LR escape evaluation queue is already active" >&2
  exit 1
fi
exec 8>"runs/dual_bbox_lr_escape_20260821.lock"
flock 8
bash evaluate_dual_bbox_lr_escape_public_20260821.sh
