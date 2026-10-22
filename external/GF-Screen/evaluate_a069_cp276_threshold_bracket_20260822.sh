#!/usr/bin/env bash
set -euo pipefail

# Public-only Focus-threshold bracket for the promising alpha=0.69 checkpoint
# 276. Glance and all postprocessing are frozen to the champion recipe.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_a069_bbox_pu_lr3e5_5ep_20260822/checkpoint_epoch_0276.pt"
sparse="${root}/public_sweeps/gf_a069_sparse_trajectory_20260822/epoch_276"
output="${root}/public_sweeps/gf_a069_cp276_threshold_bracket_20260822"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

for required in "${checkpoint}" "${sparse}/metrics.csv" "${sparse}/vs_champion.robust.json" \
    "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${output}/raw"
exec 9>"${output}.lock"
flock -n 9 || { echo "alpha-0.69 checkpoint-276 threshold audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while :; do
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used1}"
  sleep 30
done

CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
  --test_data_path "${images}" --save_prediction_path "${output}/raw" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.50 --lesion_thresholds 0.40,0.45,0.55,0.60 \
  --focus_batch_size 16 --workers 2 --skip_existing \
  > "${output}/inference.log" 2>&1

printf 'threshold\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"
for threshold in 0.40 0.45 0.55 0.60; do
  threshold_dir="lesion_$(printf '%.2f' "${threshold}")"
  tag=${threshold/./p}
  raw="${output}/raw/${threshold_dir}"
  state="${output}/l${tag}"
  filtered="${state}/filtered_mm1000"
  prompted="${state}/prompted_mm1000"
  deployed="${state}/deployed_mm1000_new2k_s25"
  mkdir -p "${filtered}" "${prompted}" "${deployed}"
  [[ $(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
    --pred-a "${raw}" --fusion a --min-mm3 1000 --workers 4 \
    --output "${filtered}" > "${state}/filter.log" 2>&1
  "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
    --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
    --output "${prompted}" > "${state}/prompted.log" 2>&1
  "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
    --base "${prompted}" --candidate "${guided_candidate}" \
    --support "${prompt_support}" --output "${deployed}" \
    --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
    --min-support-fraction 0.25 --min-support-mm3 100 \
    --support-dilation-mm inf --workers 4 > "${state}/deployed.log" 2>&1
  metrics="${state}/metrics.csv"
  robust="${state}/vs_champion.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
    > "${state}/score.log" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "a069_ep276_l${tag}" \
    --output "${robust}" > "${state}/robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${state}/score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${state}/score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${threshold}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}" \
    >> "${output}/summary.tsv"
done

dsc=$(awk '/DSC  mean/ {print $3}' "${sparse}/score.log")
nsd=$(awk '/NSD  mean/ {print $3}' "${sparse}/score.log")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${sparse}/vs_champion.robust.json")
printf '0.50\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${dsc}" "${nsd}" "${combined}" "${sparse}/metrics.csv" \
  "${sparse}/vs_champion.robust.json" "${decision}" >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k2,2gr -k3,3gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"
