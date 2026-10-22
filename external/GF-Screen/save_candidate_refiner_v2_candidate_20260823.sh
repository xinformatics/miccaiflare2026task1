#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden inference for a publicly robust bbox+24 tiled refiner.
# Hidden labels do not exist in this workflow, and this script never uploads.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${root}/public_sweeps/candidate_refiner_v2_bbox24_20260822"
selection="${root}/analysis/candidate_refiner_v2_training_20260822/checkpoint_selection.json"
saved_root="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_root="${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823"
hidden_proposals="${hidden_root}/epoch105_abd_g090_l050"
hidden_base="${hidden_root}/champion_base"
hidden_work="${hidden_root}/work_all_bbox24"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
gf_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
packager="${project}/nnunet/package_codabench.py"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"

mkdir -p "${saved_root}" "${hidden_root}" "${hidden_base}"
exec 9>"${root}/runs/save_candidate_refiner_v2_candidate_20260823.lock"
flock -n 9 || { echo "candidate-refiner v2 saver already active" >&2; exit 1; }
[[ -s "${saved_root}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${public}/PUBLIC_ONLY_COMPLETE.txt" ]]; do
  printf '%s waiting for candidate-refiner v2 public evaluation\n' "$(date -Ins)"
  sleep 30
done
for required in "${public}/summary.tsv" "${champion_zip}" "${gf_checkpoint}" "${selection}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
folds=()
for fold in 0 1 2 3 4; do
  checkpoint="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold${fold}_20260823/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  folds+=("${checkpoint}")
done

selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" \
  --summary "${public}/summary.tsv")
if [[ -z "${selected}" ]]; then
  date -Ins > "${saved_root}/NO_ELIGIBLE_CANDIDATE_REFINER_V2.txt"
  date -Ins > "${saved_root}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
IFS=$'\t' read -r variant classification segmentation deployment_mode deployment dsc nsd combined metrics robust decision <<< "${selected}"

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

proposal_marker="${hidden_proposals}/PROPOSALS_COMPLETE.json"
if [[ ! -s "${proposal_marker}" ]]; then
  wait_both_gpus
  mkdir -p "${hidden_proposals}/gpu0" "${hidden_proposals}/gpu1" "${hidden_proposals}/predictions"
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" \
      --save_prediction_path "${hidden_proposals}/gpu${shard}" \
      --trained_pth "${gf_checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 \
      --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id "${shard}" \
      --skip_existing > "${hidden_proposals}/gpu${shard}.log" 2>&1 &
    eval "pid${shard}=\$!"
  done
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
  cp "${hidden_proposals}/gpu0/"*.nii.gz "${hidden_proposals}/predictions/"
  cp "${hidden_proposals}/gpu1/"*.nii.gz "${hidden_proposals}/predictions/"
  "${python_bin}" - "${hidden_images}" "${hidden_proposals}/predictions" <<'PY'
import sys
from pathlib import Path
images=Path(sys.argv[1]); predictions=Path(sys.argv[2])
expected={path.name[:-12]+".nii.gz" for path in images.glob("*_0000.nii.gz")}
observed={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=100 or observed!=expected:
    raise SystemExit(f"hidden proposal mismatch expected={len(expected)} observed={len(observed)} missing={sorted(expected-observed)[:3]} extra={sorted(observed-expected)[:3]}")
PY
  checkpoint_sha=$(sha256sum "${gf_checkpoint}" | awk '{print $1}')
  "${python_bin}" - "${proposal_marker}" "${gf_checkpoint}" "${checkpoint_sha}" <<'PY'
import datetime,json,sys
from pathlib import Path
marker,checkpoint,digest=map(str,sys.argv[1:])
Path(marker).write_text(json.dumps({
    "created_at":datetime.datetime.now().astimezone().isoformat(),
    "cases":100,"labels_read":False,"checkpoint":str(Path(checkpoint).resolve()),
    "checkpoint_sha256":digest,"window_hu":[-175,250],
    "glance_threshold":0.90,"lesion_threshold":0.50,
},indent=2)+"\n",encoding="utf-8")
PY
fi

if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  unzip -nq "${champion_zip}" -d "${hidden_base}"
fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" \
    --proposal "epoch105=${hidden_proposals}/predictions" \
    --base "${hidden_base}" --work "${hidden_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${saved_root}/prepare_hidden.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT.json" --workers 4 \
  > "${saved_root}/audit_hidden.log" 2>&1

checkpoint_args=()
for checkpoint in "${folds[@]}"; do checkpoint_args+=(--checkpoint "${checkpoint}"); done
inference_matches() {
  "${python_bin}" - "${hidden_work}/INFERENCE_COMPLETE.json" "${folds[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); paths=[Path(path) for path in sys.argv[2:]]
expected=[str(path.resolve()) for path in paths]
if not marker.is_file(): raise SystemExit(1)
digests=[]
for path in paths:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): value.update(block)
    digests.append(value.hexdigest())
payload=json.loads(marker.read_text(encoding="utf-8"))
valid=(payload.get("checkpoints")==expected
       and payload.get("checkpoint_sha256")==digests
       and payload.get("contour_checkpoints",[])==[]
       and payload.get("contour_checkpoint_sha256",[])==[])
raise SystemExit(0 if valid else 1)
PY
}
wait_both_gpus
if ! inference_matches; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${saved_root}/infer_hidden.log" 2>&1
fi

ctag=${classification/./p}; stag=${segmentation/./p}
deployment_tag=${deployment:0:3}
case "${deployment_mode}" in
  refined) mode_tag=r ;;
  seed) mode_tag=s ;;
  seed_union) mode_tag=u ;;
  *) echo "invalid deployment mode: ${deployment_mode}" >&2; exit 2 ;;
esac
tag="gf_refv2_${deployment_tag}_${mode_tag}_c${ctag}_s${stag}"
prediction="${hidden_root}/${tag}"
empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
  --work "${hidden_work}" --base "${hidden_base}" --output "${prediction}" \
  "${empty_arg[@]}" --classification-threshold "${classification}" \
  --segmentation-threshold "${segmentation}" --mode "${deployment_mode}" \
  --min-seed-coverage 0 --min-refined-overlap 0 \
  --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
  > "${saved_root}/materialize_hidden.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" \
  --pred-dir "${prediction}" --output "${submission}" --expected-count 100 \
  > "${saved_root}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"
saved="${saved_root}/saved_candidate_refiner_v2_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
checkpoints=$(IFS=,; echo "${folds[*]}")
printf '1\t%s\tcandidate_refiner_v2_bbox24\t%s\t0.90\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoints}" "${segmentation}" "${dsc}" "${nsd}" "${metrics}" \
  "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_component_bg_saved_20260822/saved_glance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_cbg_intersection_saved_20260822/saved_intersection_submission.tsv" \
  --manifest "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/saved_wise_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_boxloc_saved_20260822/saved_boxloc_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_cbg_box_saved_20260822/saved_cbg_box_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_cbg_center_saved_20260822/saved_cbg_center_submission.tsv" \
  --manifest "${root}/public_sweeps/candidate_refiner_v1_saved_20260822/saved_candidate_refiner_submission.tsv" \
  --manifest "${saved}" --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${saved_root}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${saved_root}/calibrated_rank.log"
date -Ins > "${saved_root}/SAVE_ONLY_COMPLETE.txt"
