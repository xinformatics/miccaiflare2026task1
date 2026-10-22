#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/component_center_source_loader_smoke_20260826"
output="${work}/TERMINAL.json"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "component-center loader smoke already active" >&2; exit 1; }
[[ -s "${output}" ]] && exit 0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1 MPLCONFIGDIR="${work}/.matplotlib"
exec "${python_bin}" -u "${root}/smoke_component_center_source_loader.py" \
  --preprocessed "${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32" \
  --plans "${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json" \
  --dataset-json "${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/dataset.json" \
  --trainer-source "${root}/STUNetTrainerPanCancerSourceBalanced.py" \
  --output "${output}" >"${work}/run.log" 2>&1
