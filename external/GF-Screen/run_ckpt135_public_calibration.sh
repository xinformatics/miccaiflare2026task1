#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-runs/gf_wu_full_17k_lr3e4_20260812/checkpoint_epoch_0135.pt}"
image_dir="${PUBLIC_IMAGES:-../../validation/Validation-Public-Images}"
label_dir="${PUBLIC_LABELS:-../../validation/Validation-Public-Labels}"
output_root="${OUTPUT_ROOT:-public_sweeps/ckpt0135_calibration}"
evaluator="${EVALUATOR:-../../handoff_bundle/code/evaluate.py}"
thresholds="0.15,0.20,0.25,0.30,0.40,0.50"
gates="${GATES:-0.0 0.5}"

mkdir -p "${output_root}"
summary="${output_root}/summary.tsv"
echo -e "configuration\tDSC\tNSD\tgate\tlesion_threshold\tmin_mm3" > "${summary}"

for gate in ${gates}; do
  gate_tag="${gate/./}"
  config_dir="${output_root}/abdomen_g${gate_tag}"
  mkdir -p "${config_dir}/gpu0" "${config_dir}/gpu1"
  echo "[abdomen gate=${gate}] multi-threshold public inference"

  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" --save_prediction_path "${config_dir}/gpu0" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${gate}" --lesion_thresholds "${thresholds}" \
    --focus_batch_size 1 --skip_existing \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id 0 \
    > "${config_dir}/gpu0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${image_dir}" --save_prediction_path "${config_dir}/gpu1" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${gate}" --lesion_thresholds "${thresholds}" \
    --focus_batch_size 1 --skip_existing \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id 1 \
    > "${config_dir}/gpu1.log" 2>&1 &
  pid1=$!
  wait "${pid0}"
  wait "${pid1}"

  for threshold in 0.15 0.20 0.25 0.30 0.40 0.50; do
    threshold_dir="lesion_${threshold}"
    pred_dir="${config_dir}/${threshold_dir}/predictions"
    mkdir -p "${pred_dir}"
    cp "${config_dir}/gpu0/${threshold_dir}"/*.nii.gz "${pred_dir}/"
    cp "${config_dir}/gpu1/${threshold_dir}"/*.nii.gz "${pred_dir}/"
    count=$(find "${pred_dir}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
    if [[ "${count}" -ne 50 ]]; then
      echo "gate=${gate} threshold=${threshold}: expected 50 predictions, found ${count}" >&2
      exit 1
    fi

    score_log="${config_dir}/${threshold_dir}/score.log"
    "${python_bin}" -u "${evaluator}" --pred "${pred_dir}" --gt "${label_dir}" \
      --csv "${config_dir}/${threshold_dir}/metrics.csv" > "${score_log}" 2>&1
    dsc=$(awk '/^DSC  mean/ {print $3}' "${score_log}")
    nsd=$(awk '/^NSD  mean/ {print $3}' "${score_log}")
    echo -e "abdomen_g${gate_tag}_l${threshold/./}_mm0\t${dsc}\t${nsd}\t${gate}\t${threshold}\t0" \
      | tee -a "${summary}"
  done
done

awk -F '\t' '
  NR == 1 {next}
  {score=$2+$3; if (!found || score > best) {found=1; best=score; row=$0}}
  END {if (found) print row}
' "${summary}" > "${output_root}/best.tsv"
echo "Best checkpoint-135 no-component public calibration:"
cat "${output_root}/best.tsv"
