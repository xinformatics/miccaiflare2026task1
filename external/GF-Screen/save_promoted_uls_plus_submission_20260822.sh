#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden-validation materialization for the predeclared ULS+ public
# grid. A hidden case is never inferred unless the exact 50-case paired gate
# has returned PROMOTE. This script cannot upload to Codabench.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public_output="${root}/public_sweeps/uls_plus_rich20_refiner_20260822"
public_marker="${public_output}/PUBLIC_ONLY_COMPLETE.txt"
summary="${public_output}/summary_by_dsc.tsv"
hidden_images="${project}/validation/Validation-Hidden-Images"
checkpoint="${root}/runs/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281.pt"
base="${root}/predictions/gf_prompt_champion_new2k_s25"
adapter="${root}/run_uls_plus_component_refiner.py"
output="${root}/hidden_predictions/uls_plus_promoted_20260822"
decision_marker="${output}/SAVE_DECISION_COMPLETE.txt"

mkdir -p "${output}" "${project}/submissions"
exec 9>"${output}.lock"
flock -n 9 || { echo "promoted ULS+ saver is already active" >&2; exit 1; }
[[ -s "${decision_marker}" ]] && exit 0

while [[ ! -s "${public_marker}" ]]; do
  printf '%s waiting for complete ULS+ public grid\n' "$(date -Ins)"
  sleep 30
done

for required in "${summary}" "${checkpoint}" "${adapter}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${hidden_images}" "${base}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 hidden files: ${directory}" >&2
    exit 2
  }
done

# summary_by_dsc.tsv is already ordered by DSC then NSD. Only strict paired
# promotions are deployable; KEEP_ANCHOR rows are never allowed through.
selected=$(awk -F $'\t' 'NR > 1 && $10 == "PROMOTE" {print; exit}' "${summary}")
if [[ -z "${selected}" ]]; then
  printf '%s\n' 'No ULS+ grid state cleared the strict paired public gate; no hidden inference or ZIP was produced.' \
    > "${output}/NO_PROMOTED_SUBMISSION.txt"
  date -Ins > "${decision_marker}"
  exit 0
fi

IFS=$'\t' read -r variant mode seed overlap dsc nsd combined metrics robust decision <<< "${selected}"
[[ "${decision}" == "PROMOTE" ]] || { echo "selected non-promotion: ${variant}" >&2; exit 2; }
[[ "${mode}" == "only" || "${mode}" == "union" ]] || { echo "invalid mode: ${mode}" >&2; exit 2; }
[[ "${variant}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe variant: ${variant}" >&2; exit 2; }

tag="gf_uls_${variant}"
[[ ${#tag} -le 55 ]] || { echo "submission tag is too long: ${tag}" >&2; exit 2; }
candidate_root="${output}/${variant}"
proposals="${candidate_root}/proposals_l0p20"
work="${candidate_root}/work"
deployed="${candidate_root}/deployed"
submission="${project}/submissions/${tag}.zip"
mkdir -p "${proposals}" "${work}" "${deployed}"

gpu_memory_mib() {
  nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used1=$(gpu_memory_mib)
  if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used1}"
  sleep 30
done

export MPLCONFIGDIR="${work}/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ $(find "${proposals}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  if ! CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${proposals}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_threshold 0.20 \
      --focus_batch_size 16 --workers 2 --skip_existing \
      > "${candidate_root}/proposal_inference.log" 2>&1; then
    CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${proposals}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_threshold 0.20 \
      --focus_batch_size 8 --workers 2 --skip_existing \
      >> "${candidate_root}/proposal_inference.log" 2>&1
  fi
fi
[[ $(find "${proposals}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

common=(
  --images "${hidden_images}"
  --proposals "${proposals}"
  --work-dir "${work}"
  --output "${deployed}"
  --min-seed-mm3 20
  --max-seed-mm3 200000
  --max-components-per-case 128
  --preprocessing-workers 2
  --export-workers 2
)
if [[ ! -s "${work}/manifest.json" ]]; then
  "${python_bin}" -u "${adapter}" --stage prepare "${common[@]}" \
    > "${candidate_root}/prepare.log" 2>&1
fi
manifest_cases=$("${python_bin}" -c \
  'import json,sys; print(len(set(x["case"] for x in json.load(open(sys.argv[1])))))' \
  "${work}/manifest.json")
[[ "${manifest_cases}" -le 100 ]] || { echo "hidden manifest has invalid case count: ${manifest_cases}" >&2; exit 2; }
expected=$("${python_bin}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
  "${work}/manifest.json")

if [[ ! -s "${work}/INFERENCE_COMPLETE.txt" ]]; then
  if [[ "${expected}" -gt 0 ]]; then
    CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${adapter}" --stage infer "${common[@]}" \
      --device cuda > "${candidate_root}/uls_inference.log" 2>&1
  else
    mkdir -p "${work}/predicted_crops"
    printf '%s\n' 'No eligible hidden proposal components; GPU inference skipped.' \
      > "${candidate_root}/uls_inference.log"
  fi
  predicted=$(find "${work}/predicted_crops" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${predicted}" -eq "${expected}" ]] || {
    echo "hidden ULS+ predictions incomplete: ${predicted}/${expected}" >&2
    exit 2
  }
  date -Ins > "${work}/INFERENCE_COMPLETE.txt"
fi

materialize_args=(
  --stage materialize
  "${common[@]}"
  --min-seed-coverage "${seed}"
  --min-refined-overlap "${overlap}"
  --min-volume-ratio 0.1
  --max-volume-ratio 10
)
[[ "${mode}" == "union" ]] && materialize_args+=(--base "${base}")
"${python_bin}" -u "${adapter}" "${materialize_args[@]}" \
  > "${candidate_root}/materialize.log" 2>&1
[[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${deployed}" \
  --output "${submission}" --expected-count 100 \
  > "${candidate_root}/package.log" 2>&1
unzip -t "${submission}" > "${candidate_root}/zip_test.log"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf 'rank\ttag\tfamily\tpublic_DSC\tpublic_NSD\tmetrics\trobust\tsubmission\tsha256\tmembers\n' \
  > "${output}/saved_submission.tsv"
printf '1\t%s\tuls_plus_%s\t%s\t%s\t%s\t%s\t%s\t%s\t100\n' \
  "${tag}" "${mode}" "${dsc}" "${nsd}" "${metrics}" "${robust}" \
  "${submission}" "${digest}" >> "${output}/saved_submission.tsv"
date -Ins > "${decision_marker}"
echo "saved promoted ULS+ submission: ${submission}"
