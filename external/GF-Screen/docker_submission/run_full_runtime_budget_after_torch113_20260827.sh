#!/usr/bin/env bash
set -euo pipefail

# Join the three immutable, public-label-free compatibility replays into one
# per-case runtime/VRAM budget. This controller performs no inference and has no
# access to public labels, hidden inputs, submission packaging, or upload APIs.
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project}/external/GF-Screen/docker_submission"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
biomed="${root}/analysis/biomedparse_torch113_public_equivalence_20260827/EQUIVALENCE.json"
gf="${root}/analysis/gf_torch113_public_equivalence_20260827/EQUIVALENCE.json"
stunet="${root}/analysis/stunet_torch113_sparse_equivalence_20260827/EQUIVALENCE.json"
output_dir="${root}/analysis/full_runtime_budget_torch113_20260827"
terminal="${output_dir}/TERMINAL.json"

mkdir -p "${output_dir}"
exec 9>"${output_dir}/controller.lock"
flock -n 9 || { echo "full runtime budget controller already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE ]] && exit 0
fi

while [[ ! -s "${biomed}" || ! -s "${gf}" || ! -s "${stunet}" ]]; do
  printf '%s waiting for torch-1.13 BiomedParse/GF/STU runtime terminals\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" "${root}/summarize_runtime_budget.py" \
  --biomedparse "${biomed}" \
  --gf "${gf}" \
  --stunet "${stunet}" \
  --output "${terminal}" \
  --hard-seconds 60 \
  --target-seconds 50 \
  --hard-memory-mb 16384 \
  --efficiency-memory-mb 4096
