#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
sentinel_pid=4194303
for pid in "${CHAMPION_PID:-1126312}" "${COMPONENT_PID:-1119699}"; do
  [[ "${pid}" =~ ^[0-9]+$ ]] || { echo "invalid PID: ${pid}" >&2; exit 1; }
  while kill -0 "${pid}" 2>/dev/null; do sleep 10; done
done

cd "${workspace}"
if [[ ! -s external/BiomedParse/public_sweeps/champion_topk_rescue_20260820/summary_by_combined.tsv ]]; then
  WAIT_PID="${sentinel_pid}" bash external/BiomedParse/run_champion_topk_rescue_20260820.sh
fi
if [[ ! -s external/BiomedParse/public_sweeps/component16_biomedparse_confirmation_20260820/summary_by_combined.tsv ]]; then
  WAIT_PID="${sentinel_pid}" bash external/BiomedParse/run_component16_confirmation_20260820.sh
fi
