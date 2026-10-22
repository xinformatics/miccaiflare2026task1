#!/usr/bin/env bash
set -euo pipefail

root=/mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen
work="${root}/analysis/subtractive_gate_diag_20260823"
dataset="${work}/gate_dataset_707.csv"
marker="${work}/GATE_DATASET_707_COMPLETE.json"
builder="${work}/build_gate_dataset_707.py"

mkdir -p "${work}"
exec 9>"${work}/ensure_builder.lock"
flock -n 9 || { echo "gate-dataset watchdog already active" >&2; exit 1; }

while pgrep -f '[b]uild_gate_dataset_707.py' >/dev/null 2>&1; do
  printf '%s original gate builder still active\n' "$(date -Ins)"
  sleep 60
done

# The evaluator validates and converts the terminal output of the original
# pre-marker builder. A nonempty dataset therefore means there is no work to
# restart. New builder executions write both files atomically.
[[ -s "${dataset}" || -s "${marker}" ]] && exit 0

printf '%s no completed dataset after builder exit; restarting atomic builder\n' "$(date -Ins)"
exec nice -n 19 /home/m348911/miniconda3/envs/radoncAI/bin/python -u "${builder}"
