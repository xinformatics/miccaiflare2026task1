#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
stage_pid="${STAGE_PID:-1065829}"
biomed_pid="${BIOMED_PID:-1065830}"
lesionlocator="${workspace}/external/LesionLocator/run_champion_box_refiner_audit.sh"
smoke="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_smoke"
full="${workspace}/external/LesionLocator/public_audit/champion_mm1000_fold0_notta_full"

resume_stage() {
  kill -CONT "${stage_pid}" 2>/dev/null || true
}
trap resume_stage EXIT INT TERM

kill -STOP "${stage_pid}"
echo "[$(date --iso-8601=seconds)] paused bbox parent ${stage_pid}; BiomedParse ${biomed_pid} continues"
while true; do
  state=$(ps -o stat= -p "${biomed_pid}" 2>/dev/null | awk '{print $1}')
  [[ -z "${state}" || "${state}" == Z* ]] && break
  sleep 10
done
echo "[$(date --iso-8601=seconds)] BiomedParse boundary reached; starting corrected LesionLocator smoke"

if GPU=0 SMOKE_ONLY=1 OUTPUT_ROOT="${smoke}" bash "${lesionlocator}"; then
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
    GPU=0 SMOKE_ONLY=0 OUTPUT_ROOT="${full}" bash "${lesionlocator}" || \
      echo "[$(date --iso-8601=seconds)] full LesionLocator audit failed; retaining logs"
  else
    echo "[$(date --iso-8601=seconds)] LesionLocator failed the smoke quality/runtime gate"
  fi
else
  echo "[$(date --iso-8601=seconds)] corrected LesionLocator smoke failed; retaining logs"
fi

echo "[$(date --iso-8601=seconds)] resuming bbox/factorial queue"
resume_stage
trap - EXIT INT TERM
