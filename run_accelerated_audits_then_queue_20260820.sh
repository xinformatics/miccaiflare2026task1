#!/usr/bin/env bash
set -euo pipefail

# Recovery-safe serialized driver after replacing the accidentally serial
# BiomedParse screen. Existing official predictions are resumed case-wise.
workspace="/mnt/nvme2-data/shashank/MICCAI26"
sentinel_pid=4194303
manual_smoke_pid="${MANUAL_SMOKE_PID:-1083721}"
biomed="${workspace}/external/BiomedParse/run_public_audit_before_focus_queue.sh"
biomed_marker="${workspace}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/AUDIT_COMPLETE.json"
refiner="${workspace}/external/LesionLocator/run_champion_box_refiner_audit.sh"
smoke="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_smoke"
full="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_full"
smoke_runtime_pruned="${smoke}/PRUNED_RUNTIME.json"

if kill -0 "${sentinel_pid}" 2>/dev/null; then
  echo "NO_WAIT sentinel PID unexpectedly exists: ${sentinel_pid}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] starting accelerated BiomedParse audit"
cd "${workspace}/external/BiomedParse"
bash "${biomed}"
[[ -s "${biomed_marker}" ]] || {
  echo "BiomedParse exited without its completion marker" >&2
  exit 1
}

# The independently launched corrected smoke may still be exporting. Wait for
# that exact process, then reuse its completion guard instead of duplicating it.
while kill -0 "${manual_smoke_pid}" 2>/dev/null; do sleep 10; done
if [[ -s "${smoke_runtime_pruned}" ]]; then
  echo "[$(date --iso-8601=seconds)] LesionLocator already pruned by the prespecified runtime gate"
elif [[ ! -s "${smoke}/summary_by_combined.tsv" || ! -s "${smoke}/inference_seconds.txt" ]]; then
  echo "[$(date --iso-8601=seconds)] corrected smoke incomplete; retrying once"
  GPU=0 SMOKE_ONLY=1 OUTPUT_ROOT="${smoke}" bash "${refiner}"
fi

if [[ -s "${smoke_runtime_pruned}" ]]; then
  echo "[$(date --iso-8601=seconds)] skipping full LesionLocator audit"
else
  anchor_combined=$(awk -F $'\t' '$1 == "anchor" {print $5}' "${smoke}/summary.tsv")
  best_refiner_combined=$(awk -F $'\t' \
    '$1 != "mode" && $1 != "anchor" {if ($5 > best) best=$5} END {print best+0}' \
    "${smoke}/summary.tsv")
  seconds=$(<"${smoke}/inference_seconds.txt")
  mean_seconds=$(awk -v seconds="${seconds}" 'BEGIN {printf "%.3f", seconds/5}')
  echo "[$(date --iso-8601=seconds)] smoke anchor=${anchor_combined} best_refiner=${best_refiner_combined} mean_seconds=${mean_seconds}"
  if awk -v candidate="${best_refiner_combined}" -v anchor="${anchor_combined}" \
      -v runtime="${mean_seconds}" 'BEGIN {exit !((candidate + 0.03 >= anchor) && runtime <= 120)}'; then
    echo "[$(date --iso-8601=seconds)] starting full 50-case LesionLocator audit"
    GPU=0 SMOKE_ONLY=0 OUTPUT_ROOT="${full}" bash "${refiner}" || \
      echo "[$(date --iso-8601=seconds)] full LesionLocator audit failed; retaining logs"
  else
    echo "[$(date --iso-8601=seconds)] LesionLocator failed smoke quality/runtime gate"
  fi
fi

echo "[$(date --iso-8601=seconds)] starting repaired downstream queue"
cd "${workspace}/external/GF-Screen"
WAIT_PID="${sentinel_pid}" SKIP_COMPONENT_STAGE=1 \
  bash ./run_repaired_downstream_queue_20260820.sh
