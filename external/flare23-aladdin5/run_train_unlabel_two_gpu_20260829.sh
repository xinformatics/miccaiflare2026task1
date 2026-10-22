#!/usr/bin/env bash
set -euo pipefail

root="/mnt/nvme2-data/shashank/MICCAI26"
code="${root}/external/flare23-aladdin5"
images="${code}/train_unlabel_flat_20260829"
output="${code}/train_unlabel_roi_tumor_conf_20260829"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
mkdir -p "${output}"

test "$(find "${images}" -maxdepth 1 -type l -xtype f | wc -l)" -eq 2856
test -f "${code}/TRAIN_UNLABEL_FLAT_20260829.json"
test -f "${root}/flare23model/model_roi.pt"
test -f "${root}/flare23model/model_fine.pt"
test -f "${root}/flare23model/model_fine2.pt"
test -x "${python_bin}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${python_bin}" "${code}/run_tumor_fine_safe.py" \
  --images "${images}" \
  --output "${output}" \
  --checkpoint "${root}/flare23model/model_fine.pt" \
  --roi-checkpoint "${root}/flare23model/model_roi.pt" \
  --organ-checkpoint "${root}/flare23model/model_fine2.pt" \
  --omit-organ-inference \
  --gpu 0 \
  --role training \
  --shard-index 0 \
  --shard-count 2 \
  --terminal "${output}/TERMINAL_SHARD0.json" \
  >"${output}/shard0.log" 2>&1 &
pid0=$!

"${python_bin}" "${code}/run_tumor_fine_safe.py" \
  --images "${images}" \
  --output "${output}" \
  --checkpoint "${root}/flare23model/model_fine.pt" \
  --roi-checkpoint "${root}/flare23model/model_roi.pt" \
  --organ-checkpoint "${root}/flare23model/model_fine2.pt" \
  --omit-organ-inference \
  --gpu 1 \
  --role training \
  --shard-index 1 \
  --shard-count 2 \
  --terminal "${output}/TERMINAL_SHARD1.json" \
  >"${output}/shard1.log" 2>&1 &
pid1=$!

status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${python_bin}" "${code}/validate_train_unlabel_flare23.py" \
  --images "${images}" \
  --predictions "${output}" \
  --terminal "${output}/TERMINAL_SHARD0.json" \
  --terminal "${output}/TERMINAL_SHARD1.json" \
  --expected 2856 \
  --summary "${output}/VALIDATION_COMPLETE.json"
