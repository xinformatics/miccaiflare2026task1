#!/usr/bin/env bash
set -euo pipefail

# Factorial interaction test: component-wise trusted boxes already improved
# public DSC, while the preceding experiment changes lesion-size exposure.
# Reuse the exact sampler protocol and change only the trusted-box geometry.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"
sentinel_pid="${NO_WAIT_SENTINEL_PID:-4194303}"
if kill -0 "${sentinel_pid}" 2>/dev/null; then
  echo "NO_WAIT_SENTINEL_PID unexpectedly exists: ${sentinel_pid}" >&2
  exit 1
fi

WAIT_PID="${sentinel_pid}" \
USE_COMPONENT_BOXES=1 \
OUTPUT_ROOT="public_sweeps/gf_component_box_uniform_dual_20260820" \
UNIFORM_NAME="gf_component_box_uniform_p000_ep280_lr3e5_5ep_20260820" \
SQRT_NAME="gf_component_box_uniform_p050_ep280_lr3e5_5ep_20260820" \
bash ./run_dual_component_uniform_after_fullcoverage.sh
