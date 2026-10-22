#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt}"
image_dir="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
label_dir="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
output_root="${OUTPUT_ROOT:-public_sweeps/epoch0105}"
evaluator="${EVALUATOR:-../../handoff_bundle/code/evaluate.py}"

if [[ ! -f "${checkpoint}" ]]; then
  echo "Checkpoint does not exist: ${checkpoint}" >&2
  exit 1
fi
if [[ ! -d "${image_dir}" || ! -d "${label_dir}" ]]; then
  echo "Public images or labels are unavailable" >&2
  exit 1
fi

mkdir -p "${output_root}"
summary="${output_root}/summary.tsv"
echo -e "configuration\tDSC\tNSD" > "${summary}"

run_configuration() {
  local window_name="$1"
  local a_min="$2"
  local a_max="$3"
  local gate="$4"
  local gate_tag="${gate/./}"
  local name="${window_name}_g${gate_tag}_l05"
  local config_dir="${output_root}/${name}"
  local gpu0_dir="${config_dir}/gpu0"
  local gpu1_dir="${config_dir}/gpu1"
  local pred_dir="${config_dir}/predictions"
  mkdir -p "${gpu0_dir}" "${gpu1_dir}" "${pred_dir}"

  echo "[${name}] inference"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" \
    --save_prediction_path "${gpu0_dir}" \
    --trained_pth "${checkpoint}" \
    --a_min "${a_min}" --a_max "${a_max}" \
    --glance_threshold "${gate}" --lesion_threshold 0.5 \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
    > "${config_dir}/gpu0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" \
    --save_prediction_path "${gpu1_dir}" \
    --trained_pth "${checkpoint}" \
    --a_min "${a_min}" --a_max "${a_max}" \
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
  if [[ "${count}" -ne 50 ]]; then
    echo "${name}: expected 50 predictions, found ${count}" >&2
    exit 1
  fi

  echo "[${name}] official evaluation"
  "${python_bin}" -u "${evaluator}" \
    --pred "${pred_dir}" --gt "${label_dir}" --csv "${config_dir}/metrics.csv" \
    > "${config_dir}/score.log" 2>&1
  local dsc nsd
  dsc=$(awk '/^DSC  mean/ {print $3}' "${config_dir}/score.log")
  nsd=$(awk '/^NSD  mean/ {print $3}' "${config_dir}/score.log")
  echo -e "${name}\t${dsc}\t${nsd}" | tee -a "${summary}"
}

for gate in 0.5 0.7 0.8 0.9; do
  run_configuration abdomen -175 250 "${gate}"
done
for gate in 0.5 0.7 0.8 0.9; do
  run_configuration chest -900 650 "${gate}"
done

echo "Public sweep complete: ${summary}"
./optimize_public_postprocessing.sh "${output_root}" "${label_dir}"
./generate_hidden_from_public_best.sh "${output_root}/postprocess/best.tsv"
