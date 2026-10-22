#!/usr/bin/env bash
set -euo pipefail

# Read-only epoch-10 sentinel. It never stops, restarts, or signals training.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/nnUNetTrainerResEncS6PanCancerPartialBox__nnUNetPlans__3d_fullres_ResEncS6/fold_all"
historical="${project_root}/nnunet/RESENC_S6_HISTORICAL_PUBLIC_BASELINE_20260827.json"
auditor="${project_root}/nnunet/audit_resenc_s6_early_learning.py"
output="${project_root}/nnunet/runs/resenc_s6_partialbox_500ep_20260827/EARLY_LEARNING_EPOCH10.json"

[[ -s "${historical}" && -s "${auditor}" ]] || {
  echo "missing early-learning sentinel dependency" >&2
  exit 2
}
while [[ ! -s "${output}" ]]; do
  mapfile -t logs < <(find "${model_dir}" -maxdepth 1 -name 'training_log_*.txt' -type f | sort)
  if [[ "${#logs[@]}" -eq 1 ]] && grep -qE 'Epoch 10[[:space:]]*$' "${logs[0]}"; then
    "${python_bin}" -u "${auditor}" \
      --log "${logs[0]}" --historical "${historical}" \
      --minimum-completed-epochs 10 --output "${output}"
    exit 0
  fi
  sleep 30
done

