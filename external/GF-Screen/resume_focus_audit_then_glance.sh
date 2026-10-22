#!/usr/bin/env bash
set -euo pipefail

# Resume the 2026-08-20 calibration after all fixed-Glance predictions and
# metrics were saved. This deliberately does not repeat those 16 evaluations.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/gf_a055_bbox_ep280_focus_thresholds}"
evaluator="evaluate_public_parallel.py"
packager="../../nnunet/package_codabench.py"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${root}/summary.tsv" ]]; then
  echo "Fixed-gate summary is missing: ${root}/summary.tsv" >&2
  exit 1
fi

# Preserve only the completed fixed-gate rows so this script is idempotent.
summary_tmp=$(mktemp "${root}/summary.resume.XXXXXX")
head -1 "${root}/summary.tsv" > "${summary_tmp}"
awk -F '\t' '$1 == "0.5"' "${root}/summary.tsv" >> "${summary_tmp}"
mv "${summary_tmp}" "${root}/summary.tsv"
fixed_rows=$(awk -F '\t' '$1 == "0.5" {count++} END {print count+0}' "${root}/summary.tsv")
if [[ "${fixed_rows}" -ne 16 ]]; then
  echo "Expected 16 cached fixed-gate rows, found ${fixed_rows}" >&2
  exit 1
fi

run_low_gate_probe() {
  local gpu="$1"
  local gate="$2"
  local tag="${gate/./p}"
  local output="${root}/gate_${tag}"
  local complete=1
  for threshold in 0.50 0.60 0.70; do
    local threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
    local count=0
    if [[ -d "${output}/${threshold_dir}" ]]; then
      count=$(find "${output}/${threshold_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l)
    fi
    if [[ "${count}" -ne 50 ]]; then
      complete=0
    fi
  done
  if [[ "${complete}" -eq 1 ]]; then
    echo "Gate ${gate}: reusing complete cached predictions"
    return 0
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${gate}" --lesion_thresholds "0.50,0.60,0.70" \
    --focus_batch_size 2 --workers 2 > "${output}.inference.log" 2>&1
}

# Lower Glance thresholds are the bounded test for recovering missed lesions;
# Focus thresholds remain conservative to contain false positives.
run_low_gate_probe 0 0.20 &
low_gate_pid0=$!
run_low_gate_probe 1 0.35 &
low_gate_pid1=$!
wait "${low_gate_pid0}"
wait "${low_gate_pid1}"

score_candidate() {
  local gate="$1"
  local threshold="$2"
  local min_mm3="$3"
  local gate_tag="${gate/./p}"
  local threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  local source="${root}/gate_${gate_tag}/${threshold_dir}"
  local candidate="${root}/scored/gate_${gate_tag}_${threshold_dir}_mm${min_mm3}"
  local count
  count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "Gate ${gate}, ${threshold_dir}: expected 50 predictions, found ${count}" >&2
    return 1
  fi

  mkdir -p "${candidate}"
  if [[ "${min_mm3}" -eq 0 ]]; then
    # No need to duplicate 50 NIfTI files when no postprocessing is requested.
    candidate="${source}"
  else
    "${python_bin}" -u fuse_and_filter_masks.py \
      --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
      --output "${candidate}" > "${candidate}.filter.log" 2>&1
  fi
  local score_stem="${root}/scored/gate_${gate_tag}_${threshold_dir}_mm${min_mm3}"
  "${python_bin}" -u "${evaluator}" \
    --pred "${candidate}" --gt "${labels}" \
    --csv "${score_stem}.metrics.csv" --workers 4 > "${score_stem}.score.log" 2>&1
  local dsc nsd combined
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${gate}" "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
    >> "${root}/summary.tsv"
}

for gate in 0.20 0.35; do
  for threshold in 0.50 0.60 0.70; do
    score_candidate "${gate}" "${threshold}" 0
    score_candidate "${gate}" "${threshold}" 1000
  done
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k4,4gr
} > "${root}/summary_by_dsc.tsv"
{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k6,6gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

# Only spend hidden inference on a calibration setting that clears the paired,
# practical-effect gate against the locked champion public configuration.
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_gate best_focus best_min_mm3 best_dsc best_nsd best_combined <<< "${best_line}"
gate_tag="${best_gate/./p}"
focus_tag="${best_focus/./p}"
candidate_tag="g${gate_tag}_l${focus_tag}_mm${best_min_mm3}"
focus_dir="lesion_$(printf '%.2f' "${best_focus}")"
if [[ "${best_gate}" == "0.5" ]]; then
  selected_metrics="${root}/scored/${focus_dir}_mm${best_min_mm3}.metrics.csv"
else
  selected_metrics="${root}/scored/gate_${gate_tag}_${focus_dir}_mm${best_min_mm3}.metrics.csv"
fi
anchor_metrics="${root}/scored/lesion_0.50_mm1000.metrics.csv"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name g0p5_l0p5_mm1000 --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")

if [[ "${decision}" == "PROMOTE" ]]; then
  hidden_root="${root}/hidden_${candidate_tag}"
  mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" "${hidden_root}/filtered"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard0" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${best_gate}" --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
    > "${hidden_root}/shard0.log" 2>&1 &
  hidden_pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard1" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${best_gate}" --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 1 \
    > "${hidden_root}/shard1.log" 2>&1 &
  hidden_pid1=$!
  wait "${hidden_pid0}"
  wait "${hidden_pid1}"
  cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
  cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
  hidden_count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${hidden_count}" -ne 100 ]]; then
    echo "Expected 100 hidden predictions, found ${hidden_count}" >&2
    exit 1
  fi
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min_mm3}" \
    --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
  submission="../../submissions/gf_a055_bbox_ep280_${candidate_tag}.zip"
  "${python_bin}" -u "${packager}" \
    --images "${hidden_images}" --pred-dir "${hidden_root}/filtered" \
    --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
  unzip -t "${submission}" > "${hidden_root}/zip_test.log"
  sha256sum "${submission}" > "${submission}.sha256"
  printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
  echo "Threshold submission ready: ${submission}"
else
  printf '%s\n' "${best_line}" > "${root}/not_promoted_public_configuration.tsv"
  echo "Calibration winner did not clear the paired robustness gate; hidden inference skipped."
fi

echo "Starting Glance-only refinement."
exec ./refine_ep280_glance_only_balanced.sh
