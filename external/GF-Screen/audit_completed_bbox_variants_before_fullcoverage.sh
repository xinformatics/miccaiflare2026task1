#!/usr/bin/env bash
set -euo pipefail

# Recover two completed partial-label experiments that predate the exhaustive
# public evaluator.  They are structurally distinct from the immutable champion
# and already paid their training cost, so score one internal-best checkpoint
# from each before launching more inference experiments.
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/completed_bbox_variants_20260820}"
tmp_root="${TMP_ROOT:-/tmp/gf_completed_bbox_audit_tmp}"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
names=(consistency_ep283 component16_ep283)
checkpoints=(
  runs/gf_bbox_ep280_consistency_lr1e5_5ep_20260819/model.pt
  runs/gf_bbox_ep280_component16_lr1e5_5ep_20260819/model.pt
)

mkdir -p "${root}" ../../submissions "${tmp_root}"
export TMPDIR="${tmp_root}"
for index in 0 1; do
  name="${names[$index]}"
  checkpoint="${checkpoints[$index]}"
  [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; exit 1; }
  output="${root}/${name}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${index}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.50,0.60 \
    --focus_batch_size 2 --workers 2 > "${output}.inference.log" 2>&1 &
  if [[ "${index}" -eq 0 ]]; then pid0=$!; else pid1=$!; fi
done
wait "${pid0}"
wait "${pid1}"

printf 'variant\tcheckpoint\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
for index in 0 1; do
  name="${names[$index]}"
  checkpoint="${checkpoints[$index]}"
  for threshold in 0.50 0.60; do
    threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
    source="${root}/${name}/${threshold_dir}"
    count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
    [[ "${count}" -eq 50 ]] || {
      echo "${source}: expected 50 predictions, found ${count}" >&2
      exit 1
    }
    for min_mm3 in 0 1000; do
      score_stem="${root}/scored/${name}_${threshold_dir}_mm${min_mm3}"
      candidate="${source}"
      mkdir -p "$(dirname "${score_stem}")"
      if [[ "${min_mm3}" -gt 0 ]]; then
        candidate="${score_stem}"
        mkdir -p "${candidate}"
        "${python_bin}" -u fuse_and_filter_masks.py \
          --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
          --output "${candidate}" > "${score_stem}.filter.log" 2>&1
      fi
      "${python_bin}" -u evaluate_public_parallel.py \
        --pred "${candidate}" --gt "${public_labels}" \
        --csv "${score_stem}.metrics.csv" --workers 4 \
        > "${score_stem}.score.log" 2>&1
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${checkpoint}" "${threshold}" "${min_mm3}" \
        "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
    done
  done
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k7,7gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"
best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_name best_checkpoint best_focus best_min best_dsc best_nsd best_combined \
  <<< "${best_line}"
threshold_dir="lesion_$(printf '%.2f' "${best_focus}")"
selected_metrics="${root}/scored/${best_name}_${threshold_dir}_mm${best_min}.metrics.csv"
candidate_tag="${best_name}_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_bbox --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"

# This child script is opened only after the live Context watcher exits, unlike
# its already-running parent shell. Run the public-only oracle here unless it
# was completed while the Context parent used CPUs for scoring and left both
# GPUs idle. This makes the serialized stage restart-safe without repeating a
# label-informed diagnostic.
oracle_summary="public_sweeps/focus_oracle_public_20260820/summary_by_combined.tsv"
if [[ -s "${oracle_summary}" ]]; then
  echo "Reusing completed public Focus oracle: ${oracle_summary}" \
    > "${root}/focus_oracle_public.log"
else
  bash ./run_focus_oracle_public.sh \
    > "${root}/focus_oracle_public.log" 2>&1
fi

# Public/hidden ranking has inverted for bbox adaptation. Permit exactly one
# exploratory archive when the structurally distinct candidate is not grossly
# worse locally; this creates a file but never consumes a submission slot.
exploratory=$("${python_bin}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); ok=(float(sys.argv[2]) >= 0.465606 and d["candidate_mean"]["Lesion_DSC"] >= d["anchor_mean"]["Lesion_DSC"] - 0.02 and d["catastrophic_regression_rate"] <= 0.10); print("yes" if ok else "no")' \
  "${root}/robust_selection.json" "${best_combined}")
if [[ "${exploratory}" != "yes" ]]; then
  printf '%s\n' "${best_line}" > "${root}/exploratory_rejected.tsv"
  echo "Completed bbox candidates fell more than 0.02 combined below the anchor."
  exit 0
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
wait "${hidden_pid0}"
wait "${hidden_pid1}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
count=$(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l)
[[ "${count}" -eq 100 ]] || { echo "expected 100 hidden predictions, found ${count}" >&2; exit 1; }
if [[ "${best_min}" -gt 0 ]]; then
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min}" \
    --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
else
  cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
fi
submission="../../submissions/gf_exploratory_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/filtered" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Exploratory completed-bbox submission ready: ${submission}"
