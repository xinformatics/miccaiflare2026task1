#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
root="public_sweeps/wise_epoch105_to_275_fine"
checkpoint_root="checkpoint_interpolations/epoch105_to_275_fine"
evaluator="../../handoff_bundle/code/evaluate.py"

run_candidate() {
  local gpu="$1"
  local tag="$2"
  local checkpoint="${checkpoint_root}/wise_focus_alpha0p${tag}.pt"
  local candidate_root="${root}/wise_focus_a${tag}"
  local raw="${candidate_root}/raw"
  mkdir -p "${raw}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${raw}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_threshold 0.5 \
    --focus_batch_size 2 --workers 2 > "${candidate_root}/inference.log" 2>&1
  local count
  count=$(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "alpha ${tag}: expected 50 public predictions, found ${count}" >&2
    return 1
  fi
}

mkdir -p "${root}"
run_candidate 0 125 & pid0=$!
run_candidate 1 175 & pid1=$!
wait "${pid0}"
wait "${pid1}"
run_candidate 0 200 & pid0=$!
run_candidate 1 225 & pid1=$!
wait "${pid0}"
wait "${pid1}"
run_candidate 0 250

printf 'alpha\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
for tag in 125 175 200 225 250; do
  candidate_root="${root}/wise_focus_a${tag}"
  filtered="${candidate_root}/mm1000"
  mkdir -p "${filtered}"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${candidate_root}/raw" --fusion a --min-mm3 1000 \
    --output "${filtered}" > "${candidate_root}/filter_mm1000.log" 2>&1
  "${python_bin}" -u "${evaluator}" --pred "${filtered}" --gt "${labels}" \
    --csv "${candidate_root}/metrics_mm1000.csv" > "${candidate_root}/score_mm1000.log" 2>&1
  dsc=$(awk '/^DSC  mean/ {print $3}' "${candidate_root}/score_mm1000.log")
  nsd=$(awk '/^NSD  mean/ {print $3}' "${candidate_root}/score_mm1000.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '0.%s\t%s\t%s\t%s\n' "${tag}" "${dsc}" "${nsd}" "${combined}" \
    | tee -a "${root}/summary.tsv"
done

{ head -n 1 "${root}/summary.tsv"; tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k4,4nr; } \
  > "${root}/ranked.tsv"
cat "${root}/ranked.tsv"
