#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${project_root}/external/FreeTumor"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Hidden-Images"
checkpoint="${root}/pretrained_models/baseline/model_baseline_segmentor.pt"
checkpoint_sha="c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
output_root="${root}/hidden_predictions/independent_baseline_20260825"
output="${output_root}/raw"
mkdir -p "${output_root}" "${output}"
exec 9>"${output_root}/gpu1_controller.lock"
flock -n 9 || { echo "FreeTumor hidden controller already active" >&2;exit 1; }

[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]]
[[ $(find "${images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
for shard in $(seq 0 99); do
  shard4=$(printf '%04d' "${shard}")
  terminal="${output_root}/FREE_CASE_${shard4}.json"
  if [[ -s "${terminal}" ]] && "${python_bin}" - "${terminal}" "${checkpoint_sha}" <<'PY'
import json,sys
x=json.load(open(sys.argv[1],encoding="utf-8"))
raise SystemExit(0 if x.get("status")=="COMPLETE" and x.get("checkpoint",{}).get("sha256")==sys.argv[2] else 1)
PY
  then continue;fi
  "${python_bin}" -u "${root}/evaluate_public_safe.py" \
    --images "${images}" --output "${output}" --checkpoint "${checkpoint}" \
    --gpu 1 --shard-index "${shard}" --num-shards 100 --workers 0 \
    --sw-batch-size 1 --overlap 0.75 --dataset-role hidden --terminal "${terminal}" \
    >"${output_root}/free_case_${shard4}.log" 2>&1
done
[[ $(find "${output}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${output_root}" -maxdepth 1 -name 'FREE_CASE_*.json' | wc -l) -eq 100 ]]
