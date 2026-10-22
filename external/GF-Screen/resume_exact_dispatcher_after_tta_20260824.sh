#!/usr/bin/env bash
set -euo pipefail

# Resume only the temporarily stopped exact-707 dispatcher.  The active fold
# trainer is a child job and is deliberately never signalled by this guard.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dispatcher_pid="${EXACT_DISPATCHER_PID:-3540340}"
terminal="${root}/public_sweeps/gf_champion_focus_tta_20260824/TTA_SAVE_TERMINAL.json"
deadline_seconds="${RESUME_DEADLINE_SECONDS:-3600}"
started="$(date +%s)"

while [[ ! -s "${terminal}" ]]; do
  now="$(date +%s)"
  (( now - started < deadline_seconds )) || break
  sleep 10
done

if [[ -r "/proc/${dispatcher_pid}/cmdline" ]]; then
  command_line="$(tr '\0' ' ' < "/proc/${dispatcher_pid}/cmdline")"
  if [[ "${command_line}" == *run_candidate_refiner_v3_weak707_training_20260823.sh* ]]; then
    kill -CONT "${dispatcher_pid}"
    printf '%s resumed exact dispatcher PID %s; terminal=%s\n' \
      "$(date -Ins)" "${dispatcher_pid}" "$([[ -s "${terminal}" ]] && echo present || echo timeout)"
    exit 0
  fi
fi

printf '%s dispatcher PID %s no longer matches; no signal sent\n' \
  "$(date -Ins)" "${dispatcher_pid}"
