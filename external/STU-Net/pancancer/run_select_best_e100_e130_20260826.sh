#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
e130="${root}/public_sweeps/stunetb_matched_e130_20260826/TERMINAL.json"
output="${root}/public_sweeps/stunetb_best_e100_e130_20260826"
mkdir -p "${output}" "${root}/runs"
exec 9>"${root}/runs/select_best_e100_e130_20260826.lock"
flock -n 9 || { echo "epoch-100/130 selector already active" >&2;exit 1; }

while :; do
  if [[ ! -s "${e130}" ]]; then sleep 30;continue;fi
  status=$(/home/m348911/miniconda3/envs/radoncAI/bin/python -c \
    'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${e130}")
  case "${status}" in
    COMPLETE) break ;;
    FAILED) sleep 30 ;;
    *) echo "unexpected epoch-130 public status: ${status}" >&2;exit 2 ;;
  esac
done

/home/m348911/miniconda3/envs/radoncAI/bin/python -u \
  "${root}/select_best_e100_e130_20260826.py" \
  --e100-screen "${root}/public_sweeps/stunetb_e100_dual_prune_probability_20260826/DSC_SCREEN.json" \
  --e100-metrics "${root}/public_sweeps/stunetb_e100_anchor_confirm_either_p70_exact_20260826/metrics.csv" \
  --e100-checkpoint "${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0100.pth" \
  --e130-terminal "${e130}" --output-dir "${output}"
