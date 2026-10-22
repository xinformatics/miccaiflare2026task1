#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
work="${root}/analysis/component_center_sampling_headroom_20260826"
output="${work}/report.json"
log="${work}/run.log"
preprocessed="${project_root}/nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "component-center headroom audit already active" >&2; exit 1; }
[[ -s "${output}" ]] && exit 0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLOSC_NTHREADS=1

exec "${python_bin}" -u "${root}/analyze_component_center_sampling_headroom.py" \
  --preprocessed "${preprocessed}" --output "${output}" \
  --per-proxy-bin 12 --seed 20260826 >>"${log}" 2>&1
