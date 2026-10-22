#!/usr/bin/env bash
set -euo pipefail

selection="${1:?usage: generate_hidden_from_public_best.sh BEST_TSV}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt}"
image_dir="${HIDDEN_IMAGES:-../../validation/Validation-Hidden-Images}"
output_root="${HIDDEN_OUTPUT_ROOT:-hidden_optimized/epoch0105}"
submission="${SUBMISSION_PATH:-../../submissions/gf_screen_epoch0105_public_optimized.zip}"
packager="${PACKAGER:-../../nnunet/package_codabench.py}"

fusion=$(cut -f4 "${selection}")
source_a=$(cut -f5 "${selection}")
source_b=$(cut -f6 "${selection}")
min_mm3=$(cut -f7 "${selection}")
if [[ -z "${fusion}" || -z "${source_a}" || -z "${min_mm3}" ]]; then
  echo "Invalid public selection: ${selection}" >&2
  exit 1
fi

run_source() {
  local config="$1"
  local window="${config%%_*}"
  local remainder="${config#*_g}"
  local gate_tag="${remainder%%_*}"
  local a_min a_max gate
  case "${window}" in
    abdomen) a_min=-175; a_max=250 ;;
    chest) a_min=-900; a_max=650 ;;
    *) echo "Unknown window in ${config}" >&2; exit 1 ;;
  esac
  case "${gate_tag}" in
    05) gate=0.5 ;;
    07) gate=0.7 ;;
    08) gate=0.8 ;;
    09) gate=0.9 ;;
    *) echo "Unknown gate tag in ${config}" >&2; exit 1 ;;
  esac

  local config_dir="${output_root}/${config}"
  local gpu0_dir="${config_dir}/gpu0"
  local gpu1_dir="${config_dir}/gpu1"
  local pred_dir="${config_dir}/predictions"
  mkdir -p "${gpu0_dir}" "${gpu1_dir}" "${pred_dir}"

  echo "[hidden ${config}] inference"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" --save_prediction_path "${gpu0_dir}" \
    --trained_pth "${checkpoint}" --a_min "${a_min}" --a_max "${a_max}" \
    --glance_threshold "${gate}" --lesion_threshold 0.5 \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
    > "${config_dir}/gpu0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" --save_prediction_path "${gpu1_dir}" \
    --trained_pth "${checkpoint}" --a_min "${a_min}" --a_max "${a_max}" \
    --glance_threshold "${gate}" --lesion_threshold 0.5 \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id 1 \
    > "${config_dir}/gpu1.log" 2>&1 &
  local pid1=$!
  wait "${pid0}"
  wait "${pid1}"
  cp "${gpu0_dir}"/*.nii.gz "${pred_dir}/"
  cp "${gpu1_dir}"/*.nii.gz "${pred_dir}/"
  local count
  count=$(find "${pred_dir}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
  if [[ "${count}" -ne 100 ]]; then
    echo "${config}: expected 100 hidden predictions, found ${count}" >&2
    exit 1
  fi
}

run_source "${source_a}"
if [[ -n "${source_b}" ]]; then
  run_source "${source_b}"
fi

final_dir="${output_root}/final_predictions"
command=("${python_bin}" -u fuse_and_filter_masks.py
  --pred-a "${output_root}/${source_a}/predictions"
  --fusion "${fusion}" --min-mm3 "${min_mm3}" --output "${final_dir}")
if [[ -n "${source_b}" ]]; then
  command+=(--pred-b "${output_root}/${source_b}/predictions")
fi
"${command[@]}" > "${output_root}/final_transform.log" 2>&1

"${python_bin}" -u "${packager}" --images "${image_dir}" --pred-dir "${final_dir}" \
  --output "${submission}" --expected-count 100
echo "Optimized hidden submission: ${submission}"
