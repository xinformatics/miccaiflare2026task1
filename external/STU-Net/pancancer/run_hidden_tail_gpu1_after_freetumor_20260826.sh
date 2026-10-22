#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project_root}/validation/Validation-Hidden-Images"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
checkpoint="${model}/fold_all/checkpoint_epoch_0100.pth"
checkpoint_sha="20d94b33e378db1a5d5bd6f612df8fa95b2b97bcc88abb028711374857333675"
hidden="${root}/hidden_predictions/stunetb_e100_p70_qualified_20260826"
predictions="${hidden}/stunet"
free_root="${project_root}/external/FreeTumor/hidden_predictions/independent_baseline_20260825"
mkdir -p "${hidden}" "${predictions}"
exec 9>"${hidden}/gpu1_tail_controller.lock"
flock -n 9 || { echo "STU-Net GPU-1 tail controller already active" >&2;exit 1; }

[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]]
while [[ $(find "${free_root}/raw" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -ne 100 \
      || $(find "${free_root}" -maxdepth 1 -name 'FREE_CASE_*.json' 2>/dev/null | wc -l) -ne 100 ]]; do
  sleep 30
done
while :; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  sleep 15
done

# The ascending GPU-0 stream is well below this closed tail when this service
# is launched. Reverse order maximizes separation; existing outputs remain
# overwrite-protected by the predictor.
for part_id in $(seq 99 -1 85); do
  part4=$(printf '%04d' "${part_id}")
  "${python_bin}" -u "${root}/predict_stunet_safe.py" \
    --images "${images}" --output "${predictions}" --model-dir "${model}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu 1 \
    --part-id "${part_id}" --num-parts 100 --workers 1 --save-probabilities \
    --dataset-role hidden --terminal "${hidden}/STUNET_CASE_${part4}.json" \
    >"${hidden}/stunet_tail_case_${part4}.log" 2>&1
done
