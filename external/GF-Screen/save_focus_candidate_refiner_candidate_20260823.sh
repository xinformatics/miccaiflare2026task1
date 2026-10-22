#!/usr/bin/env bash
set -euo pipefail

# Save-only exact-100 hidden inference for a strictly promoted Focus hybrid.
# Hidden labels are unavailable; this script validates, packages, hashes, and
# ranks a ZIP but contains no upload command.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${FOCUS_SAVE_PUBLIC_ROOT:-${root}/public_sweeps/focus_candidate_refiner_20260823}"
public_terminal="${FOCUS_SAVE_PUBLIC_TERMINAL:-${public}/FOCUS_PUBLIC_TERMINAL.json}"
selection="${root}/analysis/candidate_refiner_v2_training_20260822/checkpoint_selection.json"
focus_patch_oof="${FOCUS_SAVE_PATCH_OOF:-${root}/analysis/focus_candidate_refiner_fivefold_20260823/focus_patch_oof.json}"
saved_root="${FOCUS_SAVE_ROOT:-${root}/public_sweeps/focus_candidate_refiner_saved_20260823}"
hidden_images="${project}/validation/Validation-Hidden-Images"
shared_hidden="${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823"
epoch105_root="${shared_hidden}/epoch105_abd_g090_l050"
epoch105_predictions="${epoch105_root}/predictions"
second_proposal="${FOCUS_SAVE_SECOND_PROPOSAL:-}"
proposal_dir="${FOCUS_SAVE_PROPOSAL_DIR:-${epoch105_predictions}}"
proposal_marker="${FOCUS_SAVE_PROPOSAL_MARKER:-${epoch105_root}/PROPOSALS_COMPLETE.json}"
proposal_name="${FOCUS_SAVE_PROPOSAL_NAME:-epoch105}"
hidden_base="${shared_hidden}/champion_base"
hidden_root="${FOCUS_SAVE_HIDDEN_ROOT:-${root}/hidden_predictions/focus_candidate_refiner_20260823}"
hidden_work="${hidden_root}/work_all_bbox24"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
gf_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
packager="${project}/nnunet/package_codabench.py"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
terminal="${FOCUS_SAVE_TERMINAL:-${saved_root}/FOCUS_SAVE_ONLY_COMPLETE.txt}"
controller_lock="${FOCUS_SAVE_LOCK:-${root}/runs/save_focus_candidate_refiner_candidate_20260823.lock}"
archive_prefix="${FOCUS_SAVE_ARCHIVE_PREFIX:-gf_reffoc}"
family="${FOCUS_SAVE_FAMILY:-focus_candidate_refiner}"

small_fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${saved_root}" "${hidden_root}" "${hidden_base}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "Focus candidate saver already active" >&2; exit 1; }
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${saved_root}/FOCUS_SAVE_FAILURE.json" "${status}" "${line}" "${project}/submissions" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
output=Path(sys.argv[1]); submissions=Path(sys.argv[4]); output.parent.mkdir(parents=True,exist_ok=True)
archives=[]
for path in sorted(submissions.glob("gf_ref*.zip")):
    archives.append({"path":str(path.resolve()),"bytes":path.stat().st_size,
                     "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"FAILED","exit_status":int(sys.argv[2]),"failed_line":sys.argv[3],
 "candidate_archives_found":archives,"upload_attempted":False}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
    date -Ins > "${terminal}"
  fi
  exit "${status}"
}
trap write_failure EXIT
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for Focus fixed-public decision\n' "$(date -Ins)"
  sleep 30
done
public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" != PROMOTE ]]; then
  date -Ins > "${saved_root}/NO_ELIGIBLE_FOCUS_REFINER.txt"
  date -Ins > "${terminal}"
  exit 0
fi
for required in "${public}/summary.tsv" "${champion_zip}" "${gf_checkpoint}" \
    "${selection}" "${focus_patch_oof}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" --summary "${public}/summary.tsv")
[[ -n "${selected}" ]] || { echo "public terminal says PROMOTE but no eligible row" >&2; exit 2; }
IFS=$'\t' read -r variant classification segmentation deployment_mode deployment dsc nsd combined metrics robust decision <<< "${selected}"
case "${deployment}" in anchor) deployment_tag=a;; standalone) deployment_tag=s;; *) exit 2;; esac
checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
mapfile -t selected_focus_paths < <("${python_bin}" - "${focus_patch_oof}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)):
    raise SystemExit("invalid Focus fold records")
for row in sorted(rows,key=lambda value:int(value["fold"])):
    print(row["checkpoint"])
PY
)
[[ "${#selected_focus_paths[@]}" -eq 5 ]] || exit 2
focus_input_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("input_mode","soft"))' "${focus_patch_oof}")
case "${focus_input_mode}" in soft|soft_seed|soft_lung_seed);; *) echo "invalid Focus input mode" >&2; exit 2;; esac
small_args=(); focus_args=(); small_paths=(); focus_paths=()
for fold in 0 1 2 3 4; do
  small="$(small_fold_dir "${fold}")/${checkpoint_name}"
  focus="${selected_focus_paths[$fold]}"
  for required in "${small}" "${focus}"; do [[ -s "${required}" ]] || exit 2; done
  small_args+=(--checkpoint "${small}"); focus_args+=(--contour-checkpoint "${focus}")
  small_paths+=("${small}"); focus_paths+=("${focus}")
done
gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
wait_both_gpus() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le 1024 && "${used1}" -le 1024 ]]; then return; fi
    printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"; sleep 30
  done
}
epoch105_marker="${epoch105_root}/PROPOSALS_COMPLETE.json"
if [[ ! -s "${epoch105_marker}" ]]; then
  wait_both_gpus
  mkdir -p "${epoch105_root}/gpu0" "${epoch105_root}/gpu1" "${epoch105_predictions}"
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${epoch105_root}/gpu${shard}" \
      --trained_pth "${gf_checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 --focus_batch_size 16 \
      --workers 2 --num_shards 2 --shard_id "${shard}" --skip_existing \
      > "${epoch105_root}/gpu${shard}.log" 2>&1 & eval "pid${shard}=\$!"
  done
  status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?; [[ "${status}" -eq 0 ]] || exit "${status}"
  cp "${epoch105_root}/gpu0/"*.nii.gz "${epoch105_predictions}/"
  cp "${epoch105_root}/gpu1/"*.nii.gz "${epoch105_predictions}/"
  "${python_bin}" - "${hidden_images}" "${epoch105_predictions}" "${gf_checkpoint}" "${epoch105_marker}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
images,predictions,checkpoint,marker=map(Path,sys.argv[1:])
expected={path.name[:-12]+".nii.gz" for path in images.glob("*_0000.nii.gz")}; observed={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=100 or observed!=expected: raise SystemExit("hidden proposal coverage mismatch")
payload={"created_at":datetime.now().astimezone().isoformat(),"cases":100,"labels_read":False,
 "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
 "window_hu":[-175,250],"glance_threshold":0.90,"lesion_threshold":0.50}
marker.write_text(json.dumps(payload,indent=2)+"\n")
PY
fi
[[ $(find "${epoch105_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

if [[ -n "${second_proposal}" ]]; then
  [[ $(find "${second_proposal}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
  hidden_manifest="${hidden_root}/hidden_manifest.json"
  "${python_bin}" - "${hidden_images}" "${hidden_manifest}" <<'PY'
import json,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz","image":str(image.resolve()),"complete_label":False})
if len(items)!=100 or len({item["name"] for item in items})!=100:
    raise SystemExit("expected exact 100 hidden images")
output.write_text(json.dumps({"description":"unlabeled hidden geometry manifest","items":items},indent=2)+"\n")
PY
  if [[ ! -s "${proposal_marker}" ]]; then
    mkdir -p "${proposal_dir}"
    "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
      --pred-a "${epoch105_predictions}" --pred-b "${second_proposal}" \
      --fusion union --min-mm3 0 --output "${proposal_dir}" --workers 4 \
      > "${saved_root}/fuse_proposal_union.log" 2>&1
    "${python_bin}" -u "${root}/audit_proposal_union.py" \
      --manifest "${hidden_manifest}" --source-a "${epoch105_predictions}" \
      --source-b "${second_proposal}" --union "${proposal_dir}" \
      --output "${proposal_marker}" --workers 4 --allow-unlabeled \
      > "${saved_root}/audit_proposal_union.log" 2>&1
  fi
  "${python_bin}" - "${proposal_marker}" "${epoch105_predictions}" "${second_proposal}" "${proposal_dir}" <<'PY'
import json,sys
from pathlib import Path
marker,a,b,union=map(Path,sys.argv[1:]); payload=json.loads(marker.read_text())
required=("exact_file_coverage","all_masks_finite_binary","all_geometries_match_source_images","all_unions_exact_voxelwise")
inventories=payload.get("inventories",{})
valid=(payload.get("cases")==100 and payload.get("labels_read") is False
       and all(payload.get(key) is True for key in required)
       and Path(inventories.get("source_a",{}).get("directory","")).resolve()==a.resolve()
       and Path(inventories.get("source_b",{}).get("directory","")).resolve()==b.resolve()
       and Path(inventories.get("union",{}).get("directory","")).resolve()==union.resolve())
raise SystemExit(0 if valid else "proposal-union audit mismatch")
PY
else
  [[ "${proposal_dir}" == "${epoch105_predictions}" ]]
  [[ -s "${proposal_marker}" ]]
fi
[[ $(find "${proposal_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then unzip -nq "${champion_zip}" -d "${hidden_base}"; fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare --images "${hidden_images}" \
    --proposal "${proposal_name}=${proposal_dir}" --base "${hidden_base}" --work "${hidden_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 > "${saved_root}/prepare_hidden.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" --work "${hidden_work}" \
  --output "${hidden_work}/AUDIT.json" --workers 4 > "${saved_root}/audit_hidden.log" 2>&1
inference_matches() {
  "${python_bin}" - "${hidden_work}/INFERENCE_COMPLETE.json" "${focus_input_mode}" "${#small_paths[@]}" "${small_paths[@]}" "${focus_paths[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); input_mode=sys.argv[2]; count=int(sys.argv[3]); values=list(map(Path,sys.argv[4:]));
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text()); small=values[:count]; focus=values[count:]
def sha(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): value.update(block)
    return value.hexdigest()
valid=(payload.get("checkpoints")==[str(p.resolve()) for p in small]
       and payload.get("checkpoint_sha256")==[sha(p) for p in small]
       and payload.get("contour_checkpoints")==[str(p.resolve()) for p in focus]
       and payload.get("contour_checkpoint_sha256")==[sha(p) for p in focus]
       and payload.get("contour_input_modes")==[input_mode]*len(focus))
raise SystemExit(0 if valid else 1)
PY
}
wait_both_gpus
if ! inference_matches; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" "${small_args[@]}" "${focus_args[@]}" --batch-size 2 --workers 2 \
    > "${saved_root}/infer_hidden.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT.json" --workers 4 \
  --require-predictions > "${saved_root}/audit_hidden_predictions.log" 2>&1
case "${deployment_mode}" in refined) mode_tag=r;; seed) mode_tag=s;; seed_union) mode_tag=u;; *) exit 2;; esac
tag="${archive_prefix}_${deployment_tag}_${mode_tag}_c${classification/./p}_s${segmentation/./p}"
prediction="${hidden_root}/${tag}"
empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize --work "${hidden_work}" \
  --base "${hidden_base}" --output "${prediction}" "${empty_arg[@]}" \
  --classification-threshold "${classification}" --segmentation-threshold "${segmentation}" \
  --mode "${deployment_mode}" --min-seed-coverage 0 --min-refined-overlap 0 \
  --min-volume-ratio 0 --max-volume-ratio 1000000000000 > "${saved_root}/materialize_hidden.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" --pred-dir "${prediction}" \
  --output "${submission}" --expected-count 100 > "${saved_root}/package.log" 2>&1
unzip -tq "${submission}"; [[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}'); printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"
saved="${saved_root}/saved_focus_candidate_refiner_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
checkpoints=$(IFS=,; echo "${focus_paths[*]}")
printf '1\t%s\t%s\t%s\t0.90\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoints}" "${segmentation}" "${dsc}" "${nsd}" "${metrics}" "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"
manifest_args=()
for candidate in \
  "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv" \
  "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv" \
  "${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv" \
  "${root}/public_sweeps/gf_glance_component_bg_saved_20260822/saved_glance_submission.tsv" \
  "${root}/public_sweeps/gf_cbg_intersection_saved_20260822/saved_intersection_submission.tsv" \
  "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/saved_wise_submission.tsv" \
  "${root}/public_sweeps/gf_glance_boxloc_saved_20260822/saved_boxloc_submission.tsv" \
  "${root}/public_sweeps/gf_glance_cbg_box_saved_20260822/saved_cbg_box_submission.tsv" \
  "${root}/public_sweeps/gf_glance_cbg_center_saved_20260822/saved_cbg_center_submission.tsv" \
  "${root}/public_sweeps/candidate_refiner_v1_saved_20260822/saved_candidate_refiner_submission.tsv" \
  "${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/saved_candidate_refiner_v2_submission.tsv" \
  "${saved}"; do [[ -s "${candidate}" ]] && manifest_args+=(--manifest "${candidate}"); done
"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" "${manifest_args[@]}" \
  --submitted "${submitted}" --output "${ready}" --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${saved_root}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" --history "${root}/codabench_history.tsv" \
  --candidates "${ready}" --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" > "${saved_root}/calibrated_rank.log"
date -Ins > "${terminal}"
