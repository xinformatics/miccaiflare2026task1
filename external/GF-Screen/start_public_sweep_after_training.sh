#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for the active GF-Screen trainer to finish..."
while pgrep -u "$(id -u)" -f '[G]F_RL_train.py' >/dev/null; do
  sleep 30
done

echo "Training finished; starting epoch-105 public inference sweep."
exec ./run_public_inference_sweep.sh
