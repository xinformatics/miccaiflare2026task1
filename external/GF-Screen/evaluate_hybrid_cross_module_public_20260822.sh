#!/usr/bin/env bash
set -euo pipefail

# Wait for the independently frozen Focus and Glance audits, merge their best
# NSD-guarded states, and evaluate exactly one cross-module candidate. This is
# public-only: it cannot infer hidden cases, create a ZIP, or upload anything.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
focus_audit="${root}/public_sweeps/gf_hybrid_complete04_08_deployed_20260822"
glance_audit="${root}/public_sweeps/gf_glance_hybrid04_lr_deployed_20260822"
output="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "hybrid cross-module audit is already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0

for marker in "${focus_audit}/PUBLIC_ONLY_COMPLETE.txt" \
    "${glance_audit}/PUBLIC_ONLY_COMPLETE.txt"; do
  while [[ ! -s "${marker}" ]]; do
    printf '%s waiting for %s\n' "$(date -Ins)" "${marker}"
    sleep 30
  done
done
for summary in "${focus_audit}/summary.tsv" "${glance_audit}/summary.tsv"; do
  [[ -s "${summary}" ]] || { echo "missing summary: ${summary}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 2
  }
done
for required in "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0)
  used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

"${python_bin}" "${root}/select_hybrid_cross_inputs.py" \
  --focus-summary "${focus_audit}/summary.tsv" \
  --glance-summary "${glance_audit}/summary.tsv" \
  --output "${output}/selection.json" > "${output}/selection.log"

json_value() {
  "${python_bin}" -c \
    'import json,sys; value=json.load(open(sys.argv[1])); print(value[sys.argv[2]][sys.argv[3]])' \
    "${output}/selection.json" "$1" "$2"
}
focus_checkpoint=$(json_value focus checkpoint)
glance_checkpoint=$(json_value glance checkpoint)
lesion_threshold=$(json_value focus threshold)
glance_threshold=$(json_value glance threshold)
merged="${output}/gf_hybrid_cross_best.pt"

"${python_bin}" "${root}/merge_focus_glance_checkpoints.py" \
  --focus "${focus_checkpoint}" --glance "${glance_checkpoint}" \
  --output "${merged}" > "${output}/merge.log"
"${python_bin}" "${root}/checkpoint_state_digest.py" "${merged}" \
  > "${output}/model_state.sha256"

mkdir -p "${output}"/{shard0,shard1,raw,filtered,prompted,deployed}
infer_shard() {
  local gpu="$1" shard="$2"
  local log="${output}/shard${shard}.log"
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${output}/shard${shard}" \
      --trained_pth "${merged}" --a_min -175 --a_max 250 \
      --glance_threshold "${glance_threshold}" \
      --lesion_threshold "${lesion_threshold}" --focus_batch_size 16 --workers 2 \
      --num_shards 2 --shard_id "${shard}" --skip_existing > "${log}" 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${output}/shard${shard}" \
      --trained_pth "${merged}" --a_min -175 --a_max 250 \
      --glance_threshold "${glance_threshold}" \
      --lesion_threshold "${lesion_threshold}" --focus_batch_size 8 --workers 2 \
      --num_shards 2 --shard_id "${shard}" --skip_existing >> "${log}" 2>&1
  fi
}
infer_shard 0 0 & p0=$!
infer_shard 1 1 & p1=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
cp "${output}/shard0"/*.nii.gz "${output}/raw/"
cp "${output}/shard1"/*.nii.gz "${output}/raw/"
[[ $(find "${output}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${output}/raw" --fusion a --min-mm3 1000 --workers 4 \
  --output "${output}/filtered" > "${output}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${output}/filtered" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${output}/prompted" \
  > "${output}/prompted.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${output}/prompted" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${output}/deployed" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${output}/deployed.log" 2>&1
[[ $(find "${output}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

metrics="${output}/metrics.csv"
robust="${output}/robust.json"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${output}/deployed" --gt "${labels}" --csv "${metrics}" --workers 4 \
  > "${output}/score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name champion_0p7006 --candidate-name gf_hybrid_cross_best \
  --output "${robust}" > "${output}/robust.log"
dsc=$(awk '/DSC  mean/ {print $3}' "${output}/score.log")
nsd=$(awk '/NSD  mean/ {print $3}' "${output}/score.log")
decision=$("${python_bin}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
printf 'variant\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tdecision\n' \
  > "${output}/summary.tsv"
printf 'gf_hybrid_cross\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${merged}" "${glance_threshold}" "${lesion_threshold}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
printf '%s\n' 'PUBLIC-ONLY EVALUATION COMPLETE: no hidden masks or ZIP generated.' \
  > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary.tsv"
