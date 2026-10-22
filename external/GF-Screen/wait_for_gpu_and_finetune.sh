#!/usr/bin/env bash
set -euo pipefail

while [[ ! -c /dev/nvidia0 || ! -c /dev/nvidia1 ]]; do
  printf '%s GPU device nodes unavailable; waiting 30 seconds\n' "$(date --iso-8601=seconds)"
  sleep 30
done

printf '%s Both GPU device nodes available; launching GF-Screen fine-tune\n' "$(date --iso-8601=seconds)"
exec ./run_local_finetune.sh
