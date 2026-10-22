#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
old_pgid="${OLD_PGID:-1089339}"
wait_before_guided_pid="${WAIT_BEFORE_GUIDED_PID:-}"
fusion="${workspace}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/direct_fusion"
log="${workspace}/accelerated_audits_then_queue_20260820.log"

screen_complete() {
  [[ -s "${fusion}/screen.json" ]]
}

while kill -0 -- "-${old_pgid}" 2>/dev/null; do
  if screen_complete; then
    echo "[$(date --iso-8601=seconds)] direct screen complete; loading cached/scoring acceleration"
    kill -TERM -- "-${old_pgid}" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 -- "-${old_pgid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${old_pgid}" 2>/dev/null || true
    cd "${workspace}"
    setsid env MANUAL_SMOKE_PID=1083721 SCORE_WORKERS=8 \
      WAIT_BEFORE_GUIDED_PID="${wait_before_guided_pid}" \
      bash run_accelerated_audits_then_queue_20260820.sh > "${log}" 2>&1 &
    echo "$!" > "${workspace}/accelerated_audits_then_queue_20260820.pid"
    echo "[$(date --iso-8601=seconds)] restarted accelerated wrapper as PID $!"
    exit 0
  fi
  sleep 10
done

if [[ -s "${workspace}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/AUDIT_COMPLETE.json" ]]; then
  echo "[$(date --iso-8601=seconds)] audit completed before boundary restart"
  exit 0
fi
echo "old wrapper exited before completing the direct screen" >&2
exit 1
