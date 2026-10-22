#!/usr/bin/env bash
set -euo pipefail

# Resume the existing automatic-text-proposal audit only after every earlier
# GPU workflow has exited.  BiomedParse masks are proposal maps; GF Focus still
# determines lesion boundaries.  This stage targets the measured post-prompt
# false-negative bottleneck without racing the downstream queue or its prompt
# follow-up for GPU memory.
main_queue_pid="${MAIN_QUEUE_PID:-1177736}"
prompt_followup_pid="${PROMPT_FOLLOWUP_PID:-1316151}"
score_workers="${SCORE_WORKERS:-4}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log_root="${root}/public_sweeps/automatic_text_proposals_20260820"
watcher_log="${log_root}/guided_focus_watcher.log"

for value in "${main_queue_pid}" "${prompt_followup_pid}" "${score_workers}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "PID and worker values must be non-negative integers" >&2
    exit 1
  }
done
[[ "${score_workers}" -ge 1 ]] || {
  echo "SCORE_WORKERS must be positive" >&2
  exit 1
}

mkdir -p "${log_root}"
printf '%s waiting for main queue PID %s and prompt follow-up PID %s\n' \
  "$(date --iso-8601=seconds)" "${main_queue_pid}" "${prompt_followup_pid}" \
  >> "${watcher_log}"

for wait_pid in "${main_queue_pid}" "${prompt_followup_pid}"; do
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep 30
  done
done

printf '%s barriers cleared; resuming BiomedParse-guided GF Focus audit\n' \
  "$(date --iso-8601=seconds)" >> "${watcher_log}"
SCORE_WORKERS="${score_workers}" \
  bash "${root}/run_public_audit_before_focus_queue.sh" \
  >> "${watcher_log}" 2>&1
printf '%s guided Focus audit exited successfully\n' \
  "$(date --iso-8601=seconds)" >> "${watcher_log}"
