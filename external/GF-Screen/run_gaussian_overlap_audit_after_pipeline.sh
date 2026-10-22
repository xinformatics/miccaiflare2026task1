#!/usr/bin/env bash
set -euo pipefail

# Queue a bounded overlap-aggregation audit after the active Glance pipeline.
# It evaluates the immutable epoch-280 champion and, when available, the best
# Glance-only checkpoint. No training weights are modified.
wait_pid="${WAIT_PID:-821967}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
public_images="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
root="${OUTPUT_ROOT:-public_sweeps/gaussian_overlap_20260820}"
champion="runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
glance_eval="public_sweeps/gf_glance_only_ep280_hn300_lr3e5_5ep_20260820/summary_by_combined.tsv"
glance_run="runs/gf_glance_only_ep280_hn300_lr3e5_5ep_20260820"
anchor_metrics="public_sweeps/gf_a055_bbox_ep280_focus_thresholds/scored/lesion_0.50_mm1000.metrics.csv"
thresholds="0.45,0.50,0.55,0.60,0.65,0.70"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

mkdir -p "${root}" ../../submissions
declare -a names=(champion)
declare -a checkpoints=("${champion}")
if [[ -f "${glance_eval}" ]]; then
  best_epoch=$(awk -F '\t' 'NR==2 {print $1}' "${glance_eval}")
  best_checkpoint=$(printf '%s/checkpoint_epoch_%04d.pt' "${glance_run}" "${best_epoch}")
  if [[ -f "${best_checkpoint}" ]]; then
    names+=("glance_ep${best_epoch}")
    checkpoints+=("${best_checkpoint}")
  fi
fi

infer_branch() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  local branch="${root}/${name}"
  mkdir -p "${branch}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${branch}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.5 --lesion_thresholds "${thresholds}" \
    --aggregation_mode gaussian --focus_batch_size 2 --workers 2 \
    > "${branch}.inference.log" 2>&1
}

infer_branch 0 "${names[0]}" "${checkpoints[0]}" &
pid0=$!
if [[ "${#names[@]}" -gt 1 ]]; then
  infer_branch 1 "${names[1]}" "${checkpoints[1]}" &
  pid1=$!
  wait "${pid0}"
  wait "${pid1}"
else
  wait "${pid0}"
fi

printf 'branch\tcheckpoint\tglance_threshold\tlesion_threshold\tmin_mm3\tDSC\tNSD\tcombined\n' \
  > "${root}/summary.tsv"
IFS=',' read -ra threshold_array <<< "${thresholds}"
for index in "${!names[@]}"; do
  name="${names[index]}"
  checkpoint="${checkpoints[index]}"
  for threshold in "${threshold_array[@]}"; do
    threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
    source="${root}/${name}/${threshold_dir}"
    count=$(find "${source}" -maxdepth 1 -name '*.nii.gz' | wc -l)
    if [[ "${count}" -ne 50 ]]; then
      echo "${name}/${threshold_dir}: expected 50 predictions, found ${count}" >&2
      exit 1
    fi
    for min_mm3 in 0 1000; do
      score_stem="${root}/scored/${name}_${threshold_dir}_mm${min_mm3}"
      candidate="${source}"
      if [[ "${min_mm3}" -gt 0 ]]; then
        candidate="${score_stem}"
        mkdir -p "${candidate}"
        "${python_bin}" -u fuse_and_filter_masks.py \
          --pred-a "${source}" --fusion a --min-mm3 "${min_mm3}" \
          --output "${candidate}" > "${score_stem}.filter.log" 2>&1
      else
        mkdir -p "$(dirname "${score_stem}")"
      fi
      "${python_bin}" -u evaluate_public_parallel.py \
        --pred "${candidate}" --gt "${public_labels}" \
        --csv "${score_stem}.metrics.csv" --workers 4 > "${score_stem}.score.log" 2>&1
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_stem}.score.log")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_stem}.score.log")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.4f", (d+n)/2}')
      printf '%s\t%s\t0.5\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${checkpoint}" "${threshold}" "${min_mm3}" \
        "${dsc}" "${nsd}" "${combined}" >> "${root}/summary.tsv"
    done
  done
done

{
  head -1 "${root}/summary.tsv"
  tail -n +2 "${root}/summary.tsv" | sort -t $'\t' -k8,8gr
} > "${root}/summary_by_combined.tsv"
cat "${root}/summary_by_combined.tsv"

best_line=$(sed -n '2p' "${root}/summary_by_combined.tsv")
IFS=$'\t' read -r best_name best_checkpoint best_gate best_focus best_min best_dsc best_nsd best_combined \
  <<< "${best_line}"
focus_dir="lesion_$(printf '%.2f' "${best_focus}")"
selected_metrics="${root}/scored/${best_name}_${focus_dir}_mm${best_min}.metrics.csv"
candidate_tag="${best_name}_gaussian_g05_l${best_focus/./p}_mm${best_min}"
"${python_bin}" -u select_checkpoint_robust.py \
  --anchor "${anchor_metrics}" --candidate "${selected_metrics}" \
  --anchor-name champion_uniform_g05_l05_mm1000 --candidate-name "${candidate_tag}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  printf '%s\n' "${best_line}" > "${root}/not_promoted.tsv"
  echo "Gaussian overlap candidate did not clear the paired robustness gate."
  exit 0
fi

hidden_root="${root}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" "${hidden_root}/filtered"
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard0" \
  --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold "${best_gate}" --lesion_threshold "${best_focus}" \
  --aggregation_mode gaussian --focus_batch_size 2 --workers 2 \
  --num_shards 2 --shard_id 0 > "${hidden_root}/shard0.log" 2>&1 &
hidden_pid0=$!
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
  --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard1" \
  --trained_pth "${best_checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold "${best_gate}" --lesion_threshold "${best_focus}" \
  --aggregation_mode gaussian --focus_batch_size 2 --workers 2 \
  --num_shards 2 --shard_id 1 > "${hidden_root}/shard1.log" 2>&1 &
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
  --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min}" \
  --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
submission="../../submissions/gf_${candidate_tag}.zip"
"${python_bin}" -u ../../nnunet/package_codabench.py \
  --images "${hidden_images}" --pred-dir "${hidden_root}/filtered" \
  --output "${submission}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf '%s\n' "${best_line}" > "${hidden_root}/selected_public_configuration.tsv"
echo "Gaussian overlap submission ready: ${submission}"
