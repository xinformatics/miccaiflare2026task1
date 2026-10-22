#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_boxloc_component_consensus_20260822"
upstream="${root}/public_sweeps/gf_glance_boxloc_saved_20260822/SAVE_ONLY_COMPLETE.txt"
output="${root}/hidden_predictions/gf_boxloc_component_consensus_20260822"
low_checkpoint="${root}/runs/gf_glance_boxloc_w005_allpartial_ep280_3ep_20260822/checkpoint_epoch_0283.pt"
high_checkpoint="${root}/runs/gf_glance_boxloc_w015_allpartial_ep280_3ep_20260822/checkpoint_epoch_0283.pt"
hidden_images="${project}/validation/Validation-Hidden-Images"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
whole="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/whole_nifti"
tumor="${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/tumor_nifti"
tag="gf_boxcons_e283"
submission="${project}/submissions/${tag}.zip"

mkdir -p "${output}"
exec 9>"${root}/runs/save_boxloc_consensus_candidate_20260822.lock"
flock -n 9 || { echo "boxloc consensus saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" || ! -s "${upstream}" ]]; do sleep 30; done
if [[ -s "${audit}/NO_BOUNDED_EXPLORATION.txt" || $(wc -l < "${audit}/selected.tsv") -le 1 ]]; then
  date -Ins > "${output}/NO_ELIGIBLE_CONSENSUS.txt"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
for file in "${low_checkpoint}" "${high_checkpoint}" "${champion_zip}"; do [[ -s "${file}" ]] || exit 2; done
for directory in "${whole}" "${tumor}"; do [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || exit 2; done
gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | cut -d' ' -f1; }
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  sleep 30
done
mkdir -p "${output}"/{champion,candidate,seed,persistence,selected}
unzip -oq "${champion_zip}" -d "${output}/champion"
infer() {
  local gpu="$1" checkpoint="$2" glance="$3" destination="$4" log="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${destination}" \
    --trained_pth "${checkpoint}" --glance_box_localization \
    --a_min -175 --a_max 250 --glance_threshold "${glance}" \
    --lesion_threshold 0.50 --focus_batch_size 16 --workers 2 --skip_existing \
    > "${log}" 2>&1
}
infer 0 "${low_checkpoint}" 0.20 "${output}/candidate" "${output}/candidate.log" & p0=$!
infer 1 "${low_checkpoint}" 0.50 "${output}/seed" "${output}/seed.log" & p1=$!
status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]
infer 0 "${high_checkpoint}" 0.35 "${output}/persistence" "${output}/persistence.log"
for directory in champion candidate seed persistence; do [[ $(find "${output}/${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || exit 2; done
config=$(awk -F '\t' 'NR==2{print $1}' "${audit}/selected.tsv")
metrics=$(awk -F '\t' 'NR==2{print $8}' "${audit}/selected.tsv")
robust=$(awk -F '\t' 'NR==2{print $9}' "${audit}/selected.tsv")
"${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" materialize \
  --base "${output}/champion" --candidate "${output}/candidate" --seed "${output}/seed" \
  --persistence "${output}/persistence" --whole "${whole}" --tumor "${tumor}" \
  --config "${config}" --output "${output}/selected" --workers 4 > "${output}/materialize.log" 2>&1
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/selected" \
  --output "${submission}" --expected-count 100 > "${output}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${output}/saved_boxloc_consensus_submission.tsv"
read -r dsc nsd decision policy < <(awk -F '\t' 'NR==2{print $2,$3,$10,$11}' "${audit}/selected.tsv")
printf '1\t%s\tboxcons\t%s+%s\tmulti\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${low_checkpoint}" "${high_checkpoint}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${policy}_${decision}" "${submission}" "${digest}" \
  >> "${output}/saved_boxloc_consensus_submission.tsv"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
