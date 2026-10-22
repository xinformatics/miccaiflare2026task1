#!/usr/bin/env bash
set -euo pipefail

# Submission-first resource guard. The exact-707 patch miner is resumable from
# per-case sidecars, so it can be paused while the transfer candidate is gated.
# Resume after the save pipeline reaches a terminal state, or after a bounded
# timeout so an unrelated transfer failure cannot leave mining paused forever.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
save_terminal="${root}/public_sweeps/candidate_refiner_union_transfer_saved_20260823/UNION_TRANSFER_SAVE_ONLY_TERMINAL.json"
marker="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/RESUME_AFTER_TRANSFER_TERMINAL.json"
miner_unit="miccai-cell707-training-20260823f.service"
replacement_unit="miccai-cell707-training-20260823i.service"
started_epoch="$(date +%s)"
deadline_epoch="$((started_epoch + 90 * 60))"
reason=""
terminal_status=""

while :; do
  if [[ -s "${save_terminal}" ]]; then
    terminal_status=$("${python_bin}" - "${save_terminal}" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
except Exception:
    print("")
PY
)
    case "${terminal_status}" in
      COMPLETE|SKIPPED|FAILED)
        reason="transfer_save_terminal"
        break
        ;;
    esac
  fi
  if (( $(date +%s) >= deadline_epoch )); then
    reason="timeout_90_minutes"
    terminal_status="${terminal_status:-MISSING}"
    break
  fi
  sleep 30
done

if systemctl --user is-active --quiet "${replacement_unit}"; then
  miner_unit="${replacement_unit}"
elif ! systemctl --user start "${miner_unit}"; then
  # Stopped transient units can be garbage-collected and therefore cannot
  # always be restarted by name. Recreate the exact same resumable controller
  # with absolute paths when that happens.
  miner_unit="${replacement_unit}"
  if ! systemctl --user start "${miner_unit}"; then
    systemd-run --user --unit="${miner_unit%.service}" \
      --property=Restart=on-failure --property=RestartSec=30s \
      --working-directory=/mnt/nvme2-data/shashank/MICCAI26 \
      /bin/bash -lc \
      'exec bash /mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen/run_candidate_refiner_v3_weak707_training_20260823.sh >> /mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen/logs/candidate_refiner_v3_weak707_training_20260823.log 2>&1'
  fi
fi
"${python_bin}" - "${marker}" "${reason}" "${terminal_status}" "${miner_unit}" <<'PY'
import datetime, json, os, sys
from pathlib import Path

output = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "reason": sys.argv[2],
    "transfer_save_status": sys.argv[3],
    "resumed_unit": sys.argv[4],
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
