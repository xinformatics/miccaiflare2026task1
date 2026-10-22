#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt}"
images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
root="${OUTPUT_ROOT:-public_sweeps/gf_a055_bbox_ep280_focus_thresholds}"
thresholds="${LESION_THRESHOLDS:-0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75}"
evaluator="evaluate_public_parallel.py"
packager="../../nnunet/package_codabench.py"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi

mkdir -p "${root}/shard0" "${root}/shard1"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${images}" --save_prediction_path "${root}/shard0" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_thresholds "${thresholds}" \
  --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
  > "${root}/shard0.log" 2>&1 &
pid0=$!

CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${images}" --save_prediction_path "${root}/shard1" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.5 --lesion_thresholds "${thresholds}" \
  --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id 1 \
  > "${root}/shard1.log" 2>&1 &
pid1=$!

wait "${pid0}"
wait "${pid1}"

printf 'glance_threshold\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' > "${root}/summary.tsv"
IFS=',' read -ra threshold_array <<< "${thresholds}"
for threshold in "${threshold_array[@]}"; do
  threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  merged="${root}/merged/${threshold_dir}"
  mkdir -p "${merged}"
  cp "${root}/shard0/${threshold_dir}"/*.nii.gz "${merged}/"
  cp "${root}/shard1/${threshold_dir}"/*.nii.gz "${merged}/"
  count=$(find "${merged}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 50 ]]; then
    echo "${threshold_dir}: expected 50 predictions, found ${count}" >&2
    exit 1
  fi

  for min_mm3 in 0 1000; do
    candidate="${root}/scored/${threshold_dir}_mm${min_mm3}"
    mkdir -p "${candidate}"
    "${python_bin}" -u fuse_and_filter_masks.py \
      --pred-a "${merged}" --fusion a --min-mm3 "${min_mm3}" \
      --output "${candidate}" > "${candidate}.filter.log" 2>&1
    score_log="${candidate}.score.log"
    "${python_bin}" -u "${evaluator}" \
      --pred "${candidate}" --gt "${labels}" \
      --csv "${candidate}.metrics.csv" --workers 4 > "${score_log}" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
    nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
    printf '0.5\t%s\t%s\t%s\t%s\t%s\n' "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
      >> "${root}/summary.tsv"
  done
done

run_low_gate_probe() {
  local gpu="$1"
  local gate="$2"
  local tag="${gate/./p}"
  local output="${root}/gate_${tag}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${gate}" --lesion_thresholds "0.50,0.60,0.70" \
    --focus_batch_size 2 --workers 2 > "${output}.inference.log" 2>&1
}

# A lower Glance threshold can recover complete misses, while a higher Focus
# threshold can reject weak anatomical false positives. Probe only two lower
# gates and three Focus thresholds to avoid an open-ended public-val search.
run_low_gate_probe 0 0.20 &
low_gate_pid0=$!
run_low_gate_probe 1 0.35 &
low_gate_pid1=$!
wait "${low_gate_pid0}"
wait "${low_gate_pid1}"

for gate in 0.20 0.35; do
  tag="${gate/./p}"
  for threshold in 0.50 0.60 0.70; do
    threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
    source="${root}/gate_${tag}/${threshold_dir}"
    count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
    if [[ "${count}" -ne 50 ]]; then
      echo "gate ${gate}, ${threshold_dir}: expected 50 predictions, found ${count}" >&2
      exit 1
    fi
    for min_mm3 in 0 1000; do
      candidate="${root}/scored/gate_${tag}_${threshold_dir}_mm${min_mm3}"
      mkdir -p "${candidate}"
      "${python_bin}" -u fuse_and_filter_masks.py \
        --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
        --output "${candidate}" > "${candidate}.filter.log" 2>&1
      score_log="${candidate}.score.log"
      "${python_bin}" -u "${evaluator}" \
        --pred "${candidate}" --gt "${labels}" \
        --csv "${candidate}.metrics.csv" --workers 4 > "${score_log}" 2>&1
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${gate}" "${threshold}" "${min_mm3}" "${dsc}" "${nsd}" "${combined}" \
        >> "${root}/summary.tsv"
    done
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

# Materialize the single prespecified local winner on hidden validation. This
# produces a ready-to-upload candidate without spending a Codabench submission.
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
if [[ "${decision}" != "PROMOTE" ]]; then
  echo "Local winner did not pass the paired robustness gate; no hidden ZIP generated."
  printf '%s\n' "${best_line}" > "${root}/not_promoted_public_configuration.tsv"
  exit 0
fi
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
sha256sum "${submission}" | tee "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Submission ready: ${submission}"
