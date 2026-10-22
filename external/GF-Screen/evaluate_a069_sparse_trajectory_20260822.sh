#!/usr/bin/env bash
set -euo pipefail

# Sparse public-only trajectory audit for the controlled alpha=0.69 run.
# Checkpoints 276 (warmup) and 278 (peak LR) are evaluated on GPU 1 while the
# main controller trains on GPU 0; checkpoint 280 is imported from the main
# controller's exact endpoint evaluation. No hidden inference or upload.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
run_dir="${root}/runs/gf_a069_bbox_pu_lr3e5_5ep_20260822"
main_output="${root}/public_sweeps/gf_a069_bbox_pu_deployed_20260822"
output="${root}/public_sweeps/gf_a069_sparse_trajectory_20260822"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

for required in "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "alpha-0.69 sparse trajectory audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpu_memory_mib() {
  nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_for_gpu1() {
  while :; do
    local used1
    used1=$(gpu_memory_mib)
    if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]]; then
      return
    fi
    printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used1}"
    sleep 30
  done
}

printf 'epoch\tcheckpoint\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${output}/summary.tsv"
for epoch in 276 278; do
  checkpoint="${run_dir}/checkpoint_epoch_0${epoch}.pt"
  while [[ ! -s "${checkpoint}" ]]; do
    printf '%s waiting for alpha-0.69 checkpoint %s\n' "$(date -Ins)" "${epoch}"
    sleep 30
  done
  wait_for_gpu1
  state="${output}/epoch_${epoch}"
  raw="${state}/raw_l0p50"
  filtered="${state}/filtered_l0p50_mm1000"
  prompted="${state}/prompted_l0p50_mm1000"
  deployed="${state}/deployed_l0p50_mm1000_new2k_s25"
  mkdir -p "${raw}" "${filtered}" "${prompted}" "${deployed}"
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${raw}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_batch_size 16 --workers 2 --skip_existing \
    > "${state}/inference.log" 2>&1
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
  "${python_bin}" "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "a069_bbox_ep${epoch}" \
    --output "${robust}" > "${state}/robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${state}/score.log")
  nsd=$(awk '/NSD  mean/ {print $3}' "${state}/score.log")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${epoch}" "${checkpoint}" "${dsc}" "${nsd}" "${combined}" \
    "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
done

# The main controller owns endpoint inference. Import it only after its durable
# marker appears, avoiding duplicate checkpoint-280 GPU work.
while [[ ! -s "${main_output}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for alpha-0.69 endpoint public evaluation\n' "$(date -Ins)"
  sleep 30
done
checkpoint="${run_dir}/checkpoint_epoch_0280.pt"
metrics="${main_output}/metrics.csv"
robust="${main_output}/vs_champion.robust.json"
dsc=$(awk '/DSC  mean/ {print $3}' "${main_output}/score.log")
nsd=$(awk '/NSD  mean/ {print $3}' "${main_output}/score.log")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
printf '280\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${checkpoint}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}" \
  >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k3,3gr -k4,4gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"
