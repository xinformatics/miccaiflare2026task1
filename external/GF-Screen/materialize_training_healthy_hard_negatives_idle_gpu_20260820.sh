#!/usr/bin/env bash
set -euo pipefail

# Use an idle-GPU window to materialize the already-approved training-only
# healthy hard negatives. Keep TMPDIR deliberately short because PyTorch's
# multiprocessing resource-sharer uses an AF_UNIX socket with a 108-byte path
# limit on Linux.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
healthy_json="jsons/local/healthy.json"
root="healthy_hard_negative_20260820"
staged="${root}/staged_images"
pred_root="${root}/champion_predictions"
combined="${pred_root}/combined_lesion_0.50"
crops="${root}/crops_mm1000"
derived_json="jsons/local/healthy_with_champion_hardneg_mm1000.json"
short_tmp="${HARDNEG_TMPDIR:-/tmp/gf_hn_${UID}_$$}"
unsafe_marker="${root}/SOURCE_COHORT_NOT_HEALTHY"

if [[ -f "${unsafe_marker}" ]]; then
  cat "${unsafe_marker}" >&2
  exit 1
fi

mkdir -p "${root}" "${pred_root}/shard0" "${pred_root}/shard1" \
  "${combined}" "${short_tmp}" "${short_tmp}/matplotlib"
export TMPDIR="${short_tmp}"
export MPLCONFIGDIR="${short_tmp}/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${python_bin}" -u prepare_healthy_hard_negatives.py stage \
  --healthy-json "${healthy_json}" --output "${staged}" \
  > "${root}/idle_stage.log" 2>&1
stage_count=$(find "${staged}" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${stage_count}" -eq 38 ]] || {
  echo "Expected 38 staged training-healthy images, found ${stage_count}" >&2
  exit 1
}
if find "${staged}" -maxdepth 1 ! -type d ! -name '*.nii.gz' -print -quit | grep -q .; then
  echo "Non-NIfTI entry found in flat healthy inference directory" >&2
  exit 1
fi

for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${staged}" \
    --save_prediction_path "${pred_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.50 \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${pred_root}/idle_shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then pid0=$!; else pid1=$!; fi
done
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

for shard in 0 1; do
  count=$(find "${pred_root}/shard${shard}/lesion_0.50" -maxdepth 1 \
    -name '*.nii.gz' | wc -l)
  [[ "${count}" -eq 19 ]] || {
    echo "Shard ${shard}: expected 19 predictions, found ${count}" >&2
    exit 1
  }
done
cp "${pred_root}/shard0/lesion_0.50"/*.nii.gz "${combined}/"
cp "${pred_root}/shard1/lesion_0.50"/*.nii.gz "${combined}/"
prediction_count=$(find "${combined}" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${prediction_count}" -eq 38 ]] || {
  echo "Expected 38 combined predictions, found ${prediction_count}" >&2
  exit 1
}

"${python_bin}" -u prepare_healthy_hard_negatives.py build \
  --healthy-json "${healthy_json}" --predictions "${combined}" \
  --output "${crops}" --output-json "${derived_json}" \
  --min-mm3 1000 --max-components-per-case 4 \
  --spacing 1 1 3 --roi 96 96 64 --reference-healthy-repeat 100 \
  > "${root}/idle_build.log" 2>&1

"${python_bin}" - "${crops}/manifest.json" "${derived_json}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, derived_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
derived = json.loads(derived_path.read_text(encoding="utf-8"))
assert manifest["original_cases"] == 38
assert manifest["hard_negative_crops"] > 0
assert manifest["combined_cases"] == len(derived["training"])
assert len(derived["validation"]) == 1
for item in derived["training"][38:]:
    assert Path(item["image"]).is_file()
    assert Path(item["label"]).is_file()
print(json.dumps({
    "hard_negative_crops": manifest["hard_negative_crops"],
    "combined_cases": manifest["combined_cases"],
    "recommended_healthy_repeat": manifest["recommended_healthy_repeat"],
    "derived_json": str(derived_path.resolve()),
}, indent=2))
PY

echo "Idle-GPU healthy hard-negative materialization complete: ${crops}/manifest.json"
