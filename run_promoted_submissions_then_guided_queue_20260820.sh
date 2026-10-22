#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
sentinel_pid=4194303
watcher_pid="${WATCHER_PID:-1098962}"
[[ "${watcher_pid}" =~ ^[0-9]+$ ]] || { echo "WATCHER_PID must be numeric" >&2; exit 1; }
while kill -0 "${watcher_pid}" 2>/dev/null; do sleep 15; done

echo "[$(date --iso-8601=seconds)] generating promoted OR submissions"
WAIT_PID="${sentinel_pid}" bash \
  "${workspace}/external/BiomedParse/generate_promoted_or_submissions_20260820.sh"

echo "[$(date --iso-8601=seconds)] resuming guided BiomedParse audit"
cd "${workspace}/external/BiomedParse"
WAIT_BEFORE_GUIDED_PID="${sentinel_pid}" SCORE_WORKERS=8 \
  bash ./run_public_audit_before_focus_queue.sh

echo "[$(date --iso-8601=seconds)] resuming controlled downstream queue"
cd "${workspace}/external/GF-Screen"
WAIT_PID="${sentinel_pid}" SKIP_COMPONENT_STAGE=1 \
  bash ./run_repaired_downstream_queue_20260820.sh
