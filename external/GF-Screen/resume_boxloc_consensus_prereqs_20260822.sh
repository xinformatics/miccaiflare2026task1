#!/usr/bin/env bash
set -euo pipefail

# The exact-arm evaluator may stop after a negative first result. Resume only
# the two raw masks required by champion-preserving component consensus.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
audit="${root}/public_sweeps/gf_glance_boxloc_deployed_20260822"
low_arm="gf_glance_boxloc_w005_allpartial_ep280_3ep_20260822"
high_arm="gf_glance_boxloc_w015_allpartial_ep280_3ep_20260822"
low_checkpoint="${root}/runs/${low_arm}/checkpoint_epoch_0283.pt"
high_checkpoint="${root}/runs/${high_arm}/checkpoint_epoch_0283.pt"
low_tag=$(basename "$(find "${audit}/${low_arm}" -maxdepth 1 -type d -name 'checkpoint_epoch_0283_*' -print -quit)")
high_tag=$(basename "$(find "${audit}/${high_arm}" -maxdepth 1 -type d -name 'checkpoint_epoch_0283_*' -print -quit)")
low_output="${audit}/${low_arm}/${low_tag}/g0p50/inference"
high_output="${audit}/${high_arm}/${high_tag}/g0p35/inference"

[[ -n "${low_tag}" && -n "${high_tag}" && -s "${low_checkpoint}" && -s "${high_checkpoint}" ]]
mkdir -p "${low_output}" "${high_output}"
exec 9>"${root}/runs/resume_boxloc_consensus_prereqs_20260822.lock"
flock -n 9 || { echo "boxloc consensus prerequisites already active" >&2; exit 1; }

infer() {
  local gpu="$1" checkpoint="$2" glance="$3" output="$4" log="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --glance_box_localization \
    --a_min -175 --a_max 250 --glance_threshold "${glance}" \
    --lesion_threshold 0.50 --focus_batch_size 16 --workers 2 --skip_existing \
    > "${log}" 2>&1
}
infer 0 "${low_checkpoint}" 0.50 "${low_output}" "${low_output}.resume.log" & p0=$!
infer 1 "${high_checkpoint}" 0.35 "${high_output}" "${high_output}.resume.log" & p1=$!
status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]
[[ $(find "${low_output}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${high_output}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
bash "${root}/run_boxloc_consensus_queue_20260822.sh"
