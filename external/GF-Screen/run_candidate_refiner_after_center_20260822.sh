#!/usr/bin/env bash
set -euo pipefail

# Candidate-specific appearance confirmation + voxel refinement. Training uses
# only documentation-verified complete-label cohorts; public labels are read
# only by the final evaluator. No hidden inference, packaging, or upload here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
# The refiner is the only queued branch whose measured voxel oracle exceeds
# the 0.80 objective, so run it immediately after the active all-partial audit.
upstream="${root}/public_sweeps/gf_glance_boxloc_saved_20260822/SAVE_ONLY_COMPLETE.txt"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
mining_manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
mining_images="${root}/analysis/fpr_training_mining_20260822/staged_images"
work="${root}/analysis/candidate_refiner_v1_training_20260822"
train_raw="${work}/gf_raw_l050"
patches="${work}/patches_iso15_v4"
patch_audit="${patches}/AUDIT.json"
fold0="${root}/runs/candidate_refiner_v1_fold0_20260822"
fold1="${root}/runs/candidate_refiner_v1_fold1_20260822"
fold2="${root}/runs/candidate_refiner_v1_fold2_20260822"
fold3="${root}/runs/candidate_refiner_v1_fold3_20260822"
fold4="${root}/runs/candidate_refiner_v1_fold4_20260822"
public_work="${root}/public_sweeps/candidate_refiner_v1_20260822/work_iso15"
public_connected_work="${root}/public_sweeps/candidate_refiner_v1_20260822/work_connected_iso15"
public_output="${root}/public_sweeps/candidate_refiner_v1_20260822"
public_images="${project}/validation/Validation-Public-Images"
public_labels="${project}/validation/Validation-Public-Labels"
public_proposals="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
public_base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

for required in "${checkpoint}" "${mining_manifest}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${mining_images}" "${public_images}" "${public_labels}" \
  "${public_proposals}" "${public_base}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done
mkdir -p "${work}" "${train_raw}" "${fold0}" "${fold1}" "${fold2}" \
  "${fold3}" "${fold4}" "${public_output}"
exec 9>"${root}/runs/run_candidate_refiner_after_center_20260822.lock"
flock -n 9 || { echo "candidate-refiner controller already active" >&2; exit 1; }
[[ -s "${public_output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for all-partial Glance evaluation/save stage\n' "$(date -Ins)"
  sleep 30
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR="${work}/matplotlib"
export TMPDIR="/tmp/gfref_20260822"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_both_gpus() {
  while :; do
    local used0 used1
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then return; fi
    printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}

if [[ $(find "${train_raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 210 ]]; then
  wait_both_gpus
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${mining_images}" --save_prediction_path "${train_raw}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.50 --lesion_threshold 0.50 \
      --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id "${shard}" \
      --skip_existing > "${work}/proposal_shard${shard}.log" 2>&1 &
    eval "pid${shard}=\$!"
  done
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
fi
[[ $(find "${train_raw}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]

if [[ ! -s "${patches}/COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/prepare_candidate_refiner_patches.py" \
    --manifest "${mining_manifest}" --proposal "gf_l050=${train_raw}" \
    --output "${patches}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 20000 --max-negatives-per-case 64 --workers 6 \
    > "${work}/prepare_patches.log" 2>&1
fi
[[ -s "${patches}/index.csv" && -s "${patches}/summary.json" ]]
"${python_bin}" -u "${root}/audit_candidate_refiner_patches.py" \
  --index "${patches}/index.csv" --summary "${patches}/summary.json" \
  --output "${patch_audit}" --workers 4 \
  > "${work}/audit_patches.log" 2>&1
[[ -s "${patch_audit}" ]]

wait_both_gpus
train_fold() {
  local gpu="$1" fold="$2" output="$3"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
    --index "${patches}/index.csv" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 20 --batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 16 --seed 20260822 \
    --wandb-project pancancer-ct --wandb-name "candidate_refiner_v1_fold${fold}" \
    --resume \
    > "${output}/train.log" 2>&1
}
train_sequence() {
  local gpu="$1"; shift
  local fold output
  while [[ $# -ge 2 ]]; do
    fold="$1"; output="$2"; shift 2
    train_fold "${gpu}" "${fold}" "${output}"
  done
}
train_sequence 0 0 "${fold0}" 2 "${fold2}" 4 "${fold4}" & pid0=$!
train_sequence 1 1 "${fold1}" 3 "${fold3}" & pid1=$!
status=0
wait "${pid0}" || status=$?
wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

checkpoint_args=()
checkpoint_paths=()
for fold_dir in "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}"; do
  checkpoint_args+=(--checkpoint "${fold_dir}/checkpoint_best.pt")
  checkpoint_paths+=("${fold_dir}/checkpoint_best.pt")
done
oof_summary="${work}/five_fold_oof_summary.json"
oof_args=()
for fold_dir in "${fold0}" "${fold1}" "${fold2}" "${fold3}" "${fold4}"; do
  oof_args+=(--checkpoint "${fold_dir}/checkpoint_best.pt")
  oof_args+=(--prediction "${fold_dir}/validation_predictions_best.npz")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${oof_args[@]}" --output "${oof_summary}" \
  > "${work}/five_fold_oof_summary.log" 2>&1
[[ -s "${oof_summary}" ]]

inference_matches() {
  local inference_work="$1"; shift
  "${python_bin}" - "${inference_work}/INFERENCE_COMPLETE.json" "$@" <<'PY'
import json,sys
from pathlib import Path
marker=Path(sys.argv[1])
expected=[str(Path(path).resolve()) for path in sys.argv[2:]]
if not marker.is_file(): raise SystemExit(1)
actual=json.loads(marker.read_text()).get("checkpoints",[])
raise SystemExit(0 if actual==expected else 1)
PY
}

(
  flock -x 8
  if [[ ! -s "${public_work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${public_images}" --proposal "gf_l050=${public_proposals}" \
      --base "${public_base}" --only-disconnected --work "${public_work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 20000 \
      > "${public_output}/prepare_public.log" 2>&1
  fi
) 8>"${public_work}.prepare.lock"
(
  flock -x 8
  if [[ ! -s "${public_connected_work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${public_images}" --proposal "gf_l050=${public_proposals}" \
      --base "${public_base}" --only-connected --work "${public_connected_work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 20000 \
      > "${public_output}/prepare_public_connected.log" 2>&1
  fi
) 8>"${public_connected_work}.prepare.lock"
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${public_work}" --output "${public_work}/AUDIT.json" --workers 4 \
  > "${public_output}/audit_public_cache.log" 2>&1
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${public_connected_work}" \
  --output "${public_connected_work}/AUDIT.json" --workers 4 \
  > "${public_output}/audit_public_connected_cache.log" 2>&1
wait_both_gpus
if ! inference_matches "${public_work}" "${checkpoint_paths[@]}"; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${public_work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${public_output}/infer_public.log" 2>&1
fi
if ! inference_matches "${public_connected_work}" "${checkpoint_paths[@]}"; then
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${public_connected_work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${public_output}/infer_public_connected.log" 2>&1
fi

printf 'variant\tclassification_threshold\tsegmentation_threshold\tmode\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${public_output}/summary.tsv"
classification_values=$("${python_bin}" -c '
import json,sys
center=float(json.load(open(sys.argv[1]))["classification"]["threshold"])
candidates=[center,min(0.99,center+0.10),0.90,0.95]
print(" ".join(f"{value:.2f}" for value in sorted(set(round(v,2) for v in candidates))))
' "${oof_summary}")
printf '%s\n' "${classification_values}" > "${public_output}/classification_thresholds.txt"
segmentation_values=$("${python_bin}" -c '
import json,sys
value=float(json.load(open(sys.argv[1]))["segmentation"]["threshold"])
print(f"{value:.2f}")
' "${oof_summary}")
printf '%s\n' "${segmentation_values}" > "${public_output}/segmentation_thresholds.txt"

score_prediction() {
  local tag="$1" classification="$2" segmentation="$3" mode="$4" prediction="$5"
  local metrics="${prediction}.metrics.csv"
  local score_log="${prediction}.score.log"
  local robust="${prediction}.robust.json"
  local dsc nsd combined decision
  [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${prediction}" --gt "${public_labels}" --csv "${metrics}" --workers 4 \
    > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${metrics}" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "candidate_refiner_${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${mode}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" "${robust}" "${decision}" >> "${public_output}/summary.tsv"
}

for classification in ${classification_values}; do
  tag="disc_c${classification/./p}_seed"
  prediction="${public_output}/${tag}"
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${public_work}" --base "${public_base}" --output "${prediction}" \
    --classification-threshold "${classification}" --segmentation-threshold 0.50 \
    --mode seed > "${prediction}.materialize.log" 2>&1
  score_prediction "${tag}" "${classification}" 0.50 seed "${prediction}"

  for segmentation in ${segmentation_values}; do
    for mode in refined seed_union; do
      tag="disc_c${classification/./p}_s${segmentation/./p}_${mode}"
      prediction="${public_output}/${tag}"
      "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
        --work "${public_work}" --base "${public_base}" --output "${prediction}" \
        --classification-threshold "${classification}" \
        --segmentation-threshold "${segmentation}" \
        --mode "${mode}" --min-seed-coverage 0.05 --min-refined-overlap 0.05 \
        --min-volume-ratio 0.25 --max-volume-ratio 16.0 \
        > "${prediction}.materialize.log" 2>&1
      score_prediction "${tag}" "${classification}" "${segmentation}" "${mode}" "${prediction}"
    done

    tag="repl_c${classification/./p}_s${segmentation/./p}"
    prediction="${public_output}/${tag}"
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
      --work "${public_connected_work}" --base "${public_base}" --output "${prediction}" \
      --classification-threshold "${classification}" \
      --segmentation-threshold "${segmentation}" \
      --mode replace --min-seed-coverage 0.05 --min-refined-overlap 0.05 \
      --min-volume-ratio 0.25 --max-volume-ratio 16.0 --min-base-dice 0.25 \
      --min-base-seed-coverage 0.25 \
      > "${prediction}.materialize.log" 2>&1
    score_prediction "${tag}" "${classification}" "${segmentation}" replace "${prediction}"
  done
done
{
  head -1 "${public_output}/summary.tsv"
  tail -n +2 "${public_output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${public_output}/summary_by_dsc.tsv"
date -Ins > "${public_output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${public_output}/summary_by_dsc.tsv"
bash "${root}/save_candidate_refiner_candidate_20260822.sh"
