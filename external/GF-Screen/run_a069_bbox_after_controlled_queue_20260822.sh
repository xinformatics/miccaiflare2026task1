#!/usr/bin/env bash
set -euo pipefail

# Controlled alpha=0.69 replication of the bbox run that created the champion.
# It waits until the active hybrid Focus/Glance/cross submission archives are
# safely materialized, uses only GPU 0, and is public-only: this script cannot
# create a hidden prediction or ZIP.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
checkpoint="${root}/checkpoint_interpolations/epoch105_to_275_divide/wise_focus_alpha0p690.pt"
upstream_marker="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/SAVED_BACKUPS_COMPLETE.txt"
run_name="gf_a069_bbox_pu_lr3e5_5ep_20260822"
run_dir="${root}/runs/${run_name}"
output="${root}/public_sweeps/gf_a069_bbox_pu_deployed_20260822"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

for required in "${checkpoint}" "${cache_root}/PREPROCESSING_COMPLETE.json" \
    "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${run_dir}" "${output}" "${cache_root}/tmp"
exec 9>"${root}/runs/run_a069_bbox_after_controlled_queue_20260822.lock"
flock -n 9 || { echo "alpha-0.69 bbox controller already active" >&2; exit 1; }

export TMPDIR="${cache_root}/tmp"
export MPLCONFIGDIR="${cache_root}/tmp/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while [[ ! -s "${upstream_marker}" ]]; do
  printf '%s waiting for hybrid submission archive completion\n' "$(date -Ins)"
  sleep 30
done

gpu_memory_mib() {
  nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU 0: %s MiB\n' "$(date -Ins)" "${used0}"
  sleep 30
done

# Exact champion-forming objective, shortened to the first five optimizer
# epochs because the alpha=0.55 source run reached its hidden-best state there.
if [[ ! -s "${run_dir}/checkpoint_epoch_0280.pt" \
    || ! -s "${run_dir}/model_final.pt" ]]; then
  cd "${root}"
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u GF_RL_train.py \
    --checkpoint "${checkpoint}" --start_epoch_override 275 \
    --reset_optimizer --reset_best_acc --freeze_glance \
    --partial_label_teacher --bbox_partial_label --bbox_context_mm 16 \
    --pseudo_positive_threshold 0.95 --teacher_consistency_weight 0.15 \
    --data_dir . --abdomen_json jsons/local/abdomen.json \
    --chest_json jsons/local/chest.json --healthy_json jsons/local/healthy.json \
    --healthy_repeat 100 --cache_dir "${cache_root}" --logdir "${run_dir}" \
    --max_epochs 280 --val_every 5 --batch_size 8 --sw_batch_size 2 \
    --samples_per_epoch 2730 --workers 2 --optim_lr 3e-5 --warmup_epochs 3 \
    --boundary_weight 0.15 --tversky_alpha 0.3 --tversky_beta 0.7 \
    --wandb_project pancancer-ct --wandb_name "${run_name}" \
    > "${run_dir}/train.log" 2>&1
fi

trained="${run_dir}/checkpoint_epoch_0280.pt"
[[ -s "${trained}" ]] || { echo "missing trained checkpoint: ${trained}" >&2; exit 2; }

raw="${output}/raw_l0p50"
filtered="${output}/filtered_l0p50_mm1000"
prompted="${output}/prompted_l0p50_mm1000"
deployed="${output}/deployed_l0p50_mm1000_new2k_s25"
mkdir -p "${raw}" "${filtered}" "${prompted}" "${deployed}"

CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
  --test_data_path "${images}" --save_prediction_path "${raw}" \
  --trained_pth "${trained}" --a_min -175 --a_max 250 \
  --glance_threshold 0.50 --lesion_threshold 0.50 \
  --focus_batch_size 16 --workers 2 --skip_existing \
  > "${output}/inference.log" 2>&1
[[ $(find "${raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${raw}" --fusion a --min-mm3 1000 --workers 4 \
  --output "${filtered}" > "${output}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
  --output "${prompted}" > "${output}/prompted.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${prompted}" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${deployed}" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${output}/deployed.log" 2>&1
[[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]

metrics="${output}/metrics.csv"
robust="${output}/vs_champion.robust.json"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${deployed}" --gt "${labels}" --csv "${metrics}" --workers 4 \
  > "${output}/score.log" 2>&1
"${python_bin}" "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" \
  --anchor-name gf_champ_new2k_s25 --candidate-name a069_bbox_ep280 \
  --output "${robust}" > "${output}/robust.log"

dsc=$(awk '/DSC  mean/ {print $3}' "${output}/score.log")
nsd=$(awk '/NSD  mean/ {print $3}' "${output}/score.log")
decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
printf 'candidate\tcheckpoint\tDSC\tNSD\tdecision\n%s\t%s\t%s\t%s\t%s\n' \
  "a069_bbox_ep280" "${trained}" "${dsc}" "${nsd}" "${decision}" \
  > "${output}/summary.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary.tsv"
