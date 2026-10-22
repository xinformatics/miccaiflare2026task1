#!/usr/bin/env bash
set -euo pipefail

workspace="/mnt/nvme2-data/shashank/MICCAI26"
generator_pid="${GENERATOR_PID:-1146481}"
sentinel_pid=4194303
[[ "${generator_pid}" =~ ^[0-9]+$ ]] || {
  echo "GENERATOR_PID must be numeric" >&2
  exit 1
}
while kill -0 "${generator_pid}" 2>/dev/null; do sleep 15; done

primary="${workspace}/submissions/gf_prompt_or_e050_confirm_d0_norescue.zip"
secondary="${workspace}/submissions/gf_prompt_or_e050_focus075_tumor_mean_e010_mm1000_10000.zip"
for archive in "${primary}" "${secondary}"; do
  [[ -s "${archive}" ]] || {
    echo "promoted submission generator did not produce ${archive}" >&2
    exit 1
  }
  unzip -t "${archive}" >/dev/null
done
echo "[$(date --iso-8601=seconds)] promoted submissions verified"

cd "${workspace}/external/GF-Screen"
WAIT_PID="${sentinel_pid}" bash ./run_component_box_uniform_factorial_20260820.sh
[[ -s public_sweeps/gf_component_box_uniform_dual_20260820/summary_by_combined.tsv ]] || {
  echo "component-box factorial evaluation did not produce a summary" >&2
  exit 1
}
echo "[$(date --iso-8601=seconds)] component-box factorial evaluation complete"

# The exhaustive six-prompt guided audit was pruned at this boundary: the
# public GT-proposal oracle caps the current Focus at 0.597 DSC, while direct
# prompt confirmation already reaches 0.585. Prioritize improved Focus training
# instead of roughly eight redundant guided inference passes.
WAIT_PID="${sentinel_pid}" SKIP_COMPONENT_STAGE=1 \
  bash ./run_repaired_downstream_queue_20260820.sh
