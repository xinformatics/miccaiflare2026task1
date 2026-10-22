#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Public-Images"
anchor="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_inference_epoch_0160.pth"
probe="${root}/public_sweeps/stunet_component_crop_e160_m24_probe_20260827/PROBE.json"
output="${root}/public_sweeps/stunet_component_crop_e160_m24_public_20260827"
predictions="${output}/predictions"
predict_terminal="${output}/PREDICT_TERMINAL.json"
terminal="${output}/TERMINAL.json"
gpu="${COMPONENT_CROP_GPU:-0}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MPLCONFIGDIR=/tmp/mpl-component-crop-public
mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "component-crop public audit already active" >&2; exit 1; }
"${python_bin}" - "${probe}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("margin_mm")!=24.0
    or p.get("runtime_seconds",999)>20 or p.get("delta_DSC",-999)<0
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
 raise SystemExit("24 mm component-crop probe is not eligible for public audit")
PY
used=$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]] || {
  echo "GPU ${gpu} occupied (${used} MiB); audit not started" >&2; exit 3;
}
"${python_bin}" -u "${root}/predict_stunet_component_crops.py" \
  --images "${images}" --anchors "${anchor}" --output "${predictions}" \
  --model-dir "${model}" --checkpoint "$(basename "${checkpoint}")" \
  --gpu "${gpu}" --part-id 0 --num-parts 1 --workers 1 --margin-mm 24 \
  --dataset-role public --terminal "${predict_terminal}" >"${output}/predict.log" 2>&1
"${python_bin}" -u "${root}/evaluate_component_crop_public.py" \
  --cropped "${predictions}" --predict-terminal "${predict_terminal}" \
  --output "${terminal}" --workers 2 >"${output}/evaluate.log" 2>&1
