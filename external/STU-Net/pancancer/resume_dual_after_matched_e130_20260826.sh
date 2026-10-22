#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
terminal="${root}/public_sweeps/stunetb_matched_e130_saved_20260826/TERMINAL.json"
launcher="${root}/launch_stunetb_dual_after_e100_20260826.sh"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
exec 9>"${root}/runs/resume_dual_after_matched_e130_20260826.lock"
flock -n 9 || { echo "matched epoch-130 resume watcher already active" >&2;exit 1; }
while :; do
  if [[ ! -s "${terminal}" ]]; then printf '%s waiting for matched epoch-130 saver\n' "$(date -Ins)";sleep 30;continue;fi
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED) printf '%s waiting for replacement of failed matched epoch-130 saver\n' "$(date -Ins)";sleep 30 ;;
    *) echo "unexpected matched epoch-130 saver status: ${status}" >&2;exit 2 ;;
  esac
done
exec /usr/bin/bash "${launcher}"
