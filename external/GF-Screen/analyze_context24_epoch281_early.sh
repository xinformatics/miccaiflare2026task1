#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
scorer_pid="${EARLY_SCORER_PID:-934421}"
root="public_sweeps/gf_context24_dual_20260820"
early="${root}/early_epoch281"
summary="${early}/summary_by_combined.tsv"
anchor="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"

for _ in $(seq 1 360); do
  [[ -s "${summary}" ]] && break
  if ! kill -0 "${scorer_pid}" 2>/dev/null; then
    echo "Early scorer exited without producing ${summary}" >&2
    exit 1
  fi
  sleep 20
done
[[ -s "${summary}" ]] || { echo "Timed out waiting for ${summary}" >&2; exit 1; }

best=$(sed -n '2p' "${summary}")
IFS=$'\t' read -r variant threshold minimum dsc nsd combined <<< "${best}"
tag="${variant}_lesion_${threshold}_mm${minimum}"
metrics="${early}/scored/${tag}.metrics.csv"
predictions="${root}/${variant}/checkpoint_epoch_0281/lesion_${threshold}"
if [[ "${minimum}" -gt 0 ]]; then
  predictions="${early}/scored/${tag}"
fi

"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor}" --candidate "${metrics}" \
  --anchor-name champion_bbox --candidate-name "context24_epoch281_${tag}" \
  --output "${early}/robust_selection.json" > "${early}/robust_selection.log"

"${python_bin}" -u analyze_lesion_errors.py \
  --labels "${labels}" --predictions "${predictions}" \
  --output "${early}/lesion_error_best" > "${early}/lesion_error_best.log" 2>&1

printf '%s\n' "${best}" > "${early}/selected.tsv"
cat "${early}/robust_selection.log"
cat "${early}/lesion_error_best/summary.json"
