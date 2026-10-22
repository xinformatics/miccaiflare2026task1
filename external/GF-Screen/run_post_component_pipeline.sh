#!/usr/bin/env bash
set -euo pipefail

training_pid="${1:?usage: $0 TRAINING_PID}"

while kill -0 "${training_pid}" 2>/dev/null; do
  sleep 20
done

echo "Component-aware training PID ${training_pid} finished at $(date)."
echo "Starting champion Focus-threshold audit."
./run_champion_focus_threshold_audit.sh

echo "Threshold audit completed at $(date)."
echo "Starting controlled Glance-only adaptation."
./refine_ep280_glance_only_balanced.sh
