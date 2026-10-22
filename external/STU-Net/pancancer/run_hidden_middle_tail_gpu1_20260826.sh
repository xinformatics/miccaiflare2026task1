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

mkdir -p "${hidden}" "${predictions}"
exec 9>"${hidden}/gpu1_middle_tail_controller.lock"
flock -n 9 || { echo "STU-Net GPU-1 middle-tail controller already active" >&2; exit 1; }
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]]

# The first reverse stream owns 99..85. Wait for its final case terminal, then
# reuse GPU 1 for 84..75 while the slower ascending GPU-0 stream remains below
# that range. The predictor reuses complete outputs and never overwrites them.
while [[ ! -s "${hidden}/STUNET_CASE_0085.json" ]]; do sleep 15; done
while :; do
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  sleep 15
done

for part_id in $(seq 84 -1 75); do
  part4=$(printf '%04d' "${part_id}")
  "${python_bin}" -u "${root}/predict_stunet_safe.py" \
    --images "${images}" --output "${predictions}" --model-dir "${model}" \
    --checkpoint "$(basename "${checkpoint}")" --gpu 1 \
    --part-id "${part_id}" --num-parts 100 --workers 1 --save-probabilities \
    --dataset-role hidden --terminal "${hidden}/STUNET_CASE_${part4}.json" \
    >"${hidden}/stunet_middle_tail_case_${part4}.log" 2>&1
done
