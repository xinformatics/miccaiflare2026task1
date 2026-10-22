#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
cd "${workspace}/external/GF-Screen"
echo "[$(date --iso-8601=seconds)] recovering exact component-box scoring"
bash ./finish_component_box_factorial_from_manifest_20260820.sh
echo "[$(date --iso-8601=seconds)] starting pure-GF downstream queue"
WAIT_PID=4194303 SKIP_COMPONENT_STAGE=1 \
  bash ./run_repaired_downstream_queue_20260820.sh
