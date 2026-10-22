#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
component_pid="${WAIT_COMPONENT_PID:-1048806}"
sentinel_pid=4194303
lesionlocator="${workspace}/external/LesionLocator/run_champion_box_refiner_audit.sh"
smoke="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_smoke"
full="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_full"

while kill -0 "${component_pid}" 2>/dev/null; do sleep 30; done

echo "[$(date --iso-8601=seconds)] starting LesionLocator five-case smoke audit"
if GPU=0 SMOKE_ONLY=1 OUTPUT_ROOT="${smoke}" bash "${lesionlocator}"; then
  anchor_combined=$(awk -F $'\t' '$1 == "anchor" {print $5}' "${smoke}/summary.tsv")
  best_refiner_combined=$(awk -F $'\t' \
    '$1 != "mode" && $1 != "anchor" {if ($5 > best) best=$5} END {print best+0}' \
    "${smoke}/summary.tsv")
  seconds=$(<"${smoke}/inference_seconds.txt")
  mean_seconds=$(awk -v seconds="${seconds}" 'BEGIN {printf "%.3f", seconds/5}')
  echo "[$(date --iso-8601=seconds)] smoke anchor=${anchor_combined} best_refiner=${best_refiner_combined} mean_seconds=${mean_seconds}"

  # A five-case sample is only a failure gate. Accept small noise, but reject a
  # materially worse or Docker-infeasible refiner before spending a full pass.
  if awk -v candidate="${best_refiner_combined}" -v anchor="${anchor_combined}" \
      -v runtime="${mean_seconds}" 'BEGIN {exit !((candidate + 0.03 >= anchor) && runtime <= 120)}'; then
    echo "[$(date --iso-8601=seconds)] starting full 50-case LesionLocator audit"
    GPU=0 SMOKE_ONLY=0 OUTPUT_ROOT="${full}" bash "${lesionlocator}" || \
      echo "[$(date --iso-8601=seconds)] full LesionLocator audit failed; retaining logs"
  else
    echo "[$(date --iso-8601=seconds)] LesionLocator failed the smoke quality/runtime gate"
  fi
else
  echo "[$(date --iso-8601=seconds)] LesionLocator smoke audit failed; retaining logs"
fi

echo "[$(date --iso-8601=seconds)] resuming repaired downstream queue"
cd "${workspace}/external/GF-Screen"
WAIT_PID="${sentinel_pid}" SKIP_COMPONENT_STAGE=1 \
  bash ./run_repaired_downstream_queue_20260820.sh
