#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for the two Glance calibration processes to finish..."
while pgrep -f '[G]F_RL_train.py' >/dev/null; do
  sleep 30
done
echo "Glance calibration finished; starting 17K scratch training."
exec ./run_gf_scratch_17k.sh
