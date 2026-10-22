#!/usr/bin/env bash
set -euo pipefail

# Generic hidden-label-blind save-only consumer for a strictly public-promoted
# union refiner. Defaults target union-native; the transfer wrapper supplies
# matched-transfer paths. It audits a local exact-100 ZIP and never uploads.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
role="${UNION_SAVE_ROLE:-native}"
public="${UNION_SAVE_PUBLIC_ROOT:-${root}/public_sweeps/candidate_refiner_union_native_20260823}"
public_terminal="${UNION_SAVE_PUBLIC_TERMINAL:-${public}/UNION_NATIVE_PUBLIC_TERMINAL.json}"
selection="${UNION_SAVE_SELECTION:-${root}/analysis/candidate_refiner_union_native_fullvolume_oof_20260823/deployment_checkpoint_selection.json}"
calibration="${UNION_SAVE_CALIBRATION:-${root}/analysis/candidate_refiner_union_native_fullvolume_oof_20260823/deployment_calibration.json}"
saved_root="${UNION_SAVE_ROOT:-${root}/public_sweeps/candidate_refiner_union_native_saved_20260823}"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_root="${UNION_SAVE_HIDDEN_ROOT:-${root}/hidden_predictions/candidate_refiner_union_native_20260823}"
epoch105_root="${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823/epoch105_abd_g090_l050"
epoch105="${epoch105_root}/predictions"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
proposal_union="${hidden_root}/proposal_union"
hidden_base="${root}/hidden_predictions/candidate_refiner_v2_bbox24_20260823/champion_base"
hidden_work="${hidden_root}/work_all_bbox24"
champion_zip="${project}/submissions/gf_champ_new2k_s25.zip"
gf_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
packager="${project}/nnunet/package_codabench.py"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"
terminal="${UNION_SAVE_TERMINAL:-${saved_root}/UNION_NATIVE_SAVE_ONLY_TERMINAL.json}"
run_prefix="${UNION_SAVE_RUN_PREFIX:-candidate_refiner_union_native_b32_fold}"
archive_prefix="${UNION_SAVE_ARCHIVE_PREFIX:-gf_refun}"
family="${UNION_SAVE_FAMILY:-candidate_refiner_union_native}"
saved_filename="${UNION_SAVE_MANIFEST_NAME:-saved_union_native_submission.tsv}"
controller_lock="${UNION_SAVE_LOCK:-${root}/runs/save_candidate_refiner_union_native_candidate_20260823.lock}"

case "${role}" in native|transfer);; *) echo "invalid UNION_SAVE_ROLE: ${role}" >&2; exit 2;; esac
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }

mkdir -p "${saved_root}" "${hidden_root}" "${hidden_base}" "${epoch105}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "union ${role} save-only consumer already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"NO_SUBMISSION","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_labels_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  fi
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for union %s fixed-public decision\n' "$(date -Ins)" "${role}"
  sleep 30
done
public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" != PROMOTE ]]; then
  "${python_bin}" - "${public_terminal}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"NO_SUBMISSION","reason":"fixed public gate did not promote",
 "source":str(source.resolve()),"public_read":False,"hidden_labels_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
for required in "${public}/summary.tsv" "${selection}" "${calibration}" "${champion_zip}" "${gf_checkpoint}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" --summary "${public}/summary.tsv")
[[ -n "${selected}" ]] || { echo "public terminal promoted but summary has no valid promoted row" >&2; exit 2; }
IFS=$'\t' read -r variant classification segmentation deployment_mode deployment dsc nsd combined metrics robust gate <<< "${selected}"

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
contour_checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("contour_checkpoint_filename","NONE"))' \
  "${selection}")
folds=(); contour_folds=()
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  folds+=("${checkpoint}")
  if [[ "${contour_checkpoint_name}" != NONE ]]; then
    contour_checkpoint="$(fold_dir "${fold}")/${contour_checkpoint_name}"
    [[ -s "${contour_checkpoint}" ]] || { echo "missing contour checkpoint: ${contour_checkpoint}" >&2; exit 2; }
    contour_folds+=("${contour_checkpoint}")
  fi
done

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
wait_any_gpu() {
  while :; do
    local used0 used1
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]]; then selected_gpu=0; return; fi
    if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]]; then selected_gpu=1; return; fi
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}

proposal_marker="${epoch105_root}/PROPOSALS_COMPLETE.json"
if [[ ! -s "${proposal_marker}" ]]; then
  wait_any_gpu
  mkdir -p "${epoch105_root}/gpu0" "${epoch105_root}/gpu1" "${epoch105}"
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${selected_gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${epoch105_root}/gpu${shard}" \
      --trained_pth "${gf_checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 --focus_batch_size 16 \
      --workers 2 --num_shards 2 --shard_id "${shard}" --skip_existing \
      > "${saved_root}/epoch105_gpu${shard}.log" 2>&1
  done
  cp "${epoch105_root}/gpu0/"*.nii.gz "${epoch105}/"
  cp "${epoch105_root}/gpu1/"*.nii.gz "${epoch105}/"
  checkpoint_sha=$(sha256sum "${gf_checkpoint}" | awk '{print $1}')
  "${python_bin}" - "${hidden_images}" "${epoch105}" "${proposal_marker}" "${gf_checkpoint}" "${checkpoint_sha}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
images,predictions,marker,checkpoint=map(Path,sys.argv[1:5]); digest=sys.argv[5]
expected={path.name[:-12]+".nii.gz" for path in images.glob("*_0000.nii.gz")}
observed={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=100 or observed!=expected: raise SystemExit("hidden epoch105 coverage mismatch")
payload={"created_at":datetime.datetime.now().astimezone().isoformat(),"cases":100,
 "labels_read":False,"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":digest,
 "window_hu":[-175,250],"glance_threshold":0.90,"lesion_threshold":0.50}
temporary=marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,marker)
PY
fi
[[ $(find "${epoch105}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${currentgf}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

hidden_manifest="${hidden_root}/hidden_manifest.json"
"${python_bin}" - "${hidden_images}" "${hidden_manifest}" <<'PY'
import json,os,sys
from pathlib import Path
images,output=map(Path,sys.argv[1:]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz","image":str(image.resolve()),"complete_label":False})
if len(items)!=100 or len({item["name"] for item in items})!=100: raise SystemExit("expected exact 100 hidden images")
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps({"description":"unlabeled hidden geometry manifest","items":items},indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
union_marker="${hidden_root}/PROPOSAL_UNION_COMPLETE.json"
if [[ ! -s "${union_marker}" ]]; then
  mkdir -p "${proposal_union}"
  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" --pred-a "${epoch105}" \
    --pred-b "${currentgf}" --fusion union --min-mm3 0 --output "${proposal_union}" \
    --workers 4 > "${saved_root}/fuse_union.log" 2>&1
  "${python_bin}" -u "${root}/audit_proposal_union.py" --manifest "${hidden_manifest}" \
    --source-a "${epoch105}" --source-b "${currentgf}" --union "${proposal_union}" \
    --output "${union_marker}" --workers 4 --allow-unlabeled \
    > "${saved_root}/audit_union.log" 2>&1
fi
"${python_bin}" - "${union_marker}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); required=("exact_file_coverage","all_masks_finite_binary","all_geometries_match_source_images","all_unions_exact_voxelwise")
if p.get("cases")!=100 or not all(p.get(k) is True for k in required) or p.get("labels_read") is not False:
    raise SystemExit("hidden proposal-union audit failed")
PY

if [[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
  unzip -nq "${champion_zip}" -d "${hidden_base}"
fi
[[ $(find "${hidden_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" --proposal "union=${proposal_union}" --base "${hidden_base}" \
    --work "${hidden_work}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${saved_root}/prepare_hidden.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT.json" --workers 4 \
  > "${saved_root}/audit_cache.log" 2>&1

checkpoint_args=(); for checkpoint in "${folds[@]}"; do checkpoint_args+=(--checkpoint "${checkpoint}"); done
contour_checkpoint_args=(); for checkpoint in "${contour_folds[@]}"; do contour_checkpoint_args+=(--contour-checkpoint "${checkpoint}"); done
inference_matches() {
  "${python_bin}" - "${hidden_work}/INFERENCE_COMPLETE.json" 5 \
    "${folds[@]}" "${contour_folds[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); count=int(sys.argv[2])
checkpoints=list(map(Path,sys.argv[3:3+count])); contours=list(map(Path,sys.argv[3+count:]))
if not marker.is_file(): raise SystemExit(1)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
p=json.loads(marker.read_text(encoding="utf-8"))
valid=(p.get("checkpoints")==[str(path.resolve()) for path in checkpoints]
       and p.get("checkpoint_sha256")==[sha(path) for path in checkpoints]
       and p.get("contour_checkpoints",[])==[str(path.resolve()) for path in contours]
       and p.get("contour_checkpoint_sha256",[])==[sha(path) for path in contours])
raise SystemExit(0 if valid else 1)
PY
}
if ! inference_matches; then
  wait_any_gpu
  CUDA_VISIBLE_DEVICES="${selected_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" "${checkpoint_args[@]}" "${contour_checkpoint_args[@]}" \
    --batch-size 8 --workers 2 \
    > "${saved_root}/infer_hidden.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${hidden_work}" --output "${hidden_work}/AUDIT_INFERENCE.json" --workers 4 \
  --require-predictions > "${saved_root}/audit_inference.log" 2>&1

case "${deployment_mode}" in refined) mode_tag=r;; seed) mode_tag=s;; seed_union) mode_tag=u;; *) exit 2;; esac
case "${deployment}" in anchor) deployment_tag=anc;; standalone) deployment_tag=std;; *) exit 2;; esac
tag="${archive_prefix}_${deployment_tag}_${mode_tag}_c${classification/./p}_s${segmentation/./p}"
[[ ${#tag} -lt 60 ]] || { echo "submission tag is too long" >&2; exit 2; }
prediction="${hidden_root}/${tag}"; empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
  --work "${hidden_work}" --base "${hidden_base}" --output "${prediction}" "${empty_arg[@]}" \
  --classification-threshold "${classification}" --segmentation-threshold "${segmentation}" \
  --mode "${deployment_mode}" --min-seed-coverage 0 --min-refined-overlap 0 \
  --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
  > "${saved_root}/materialize.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

submission="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" --pred-dir "${prediction}" \
  --output "${submission}" --expected-count 100 > "${saved_root}/package.log" 2>&1
unzip -tq "${submission}" > "${saved_root}/zip_test.log"
member_count=$(unzip -Z1 "${submission}" | awk 'NF && $0 !~ /\/$/ {n++} END{print n+0}')
nested_count=$(unzip -Z1 "${submission}" | awk 'NF && $0 !~ /\/$/ && index($0,"/") {n++} END{print n+0}')
[[ "${member_count}" -eq 100 && "${nested_count}" -eq 0 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"

saved="${saved_root}/${saved_filename}"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
checkpoints=$(IFS=,; echo "${folds[*]}")
if [[ "${#contour_folds[@]}" -gt 0 ]]; then
  contour_checkpoints=$(IFS=,; echo "${contour_folds[*]}")
  checkpoints="${checkpoints};contours=${contour_checkpoints}"
fi
printf '1\t%s\t%s\t%s\t0.90+0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoints}" "${segmentation}" "${dsc}" "${nsd}" "${metrics}" \
  "${robust}" "${gate}" "${submission}" "${digest}" >> "${saved}"

manifest_args=()
while IFS= read -r candidate; do
  [[ -s "${candidate}" ]] && manifest_args+=(--manifest "${candidate}")
done < <(find "${root}/public_sweeps" -maxdepth 3 -type f -name 'saved*submission*.tsv' -print | sort)
"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" "${manifest_args[@]}" \
  --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" \
  > "${saved_root}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${saved_root}/calibrated_rank.log"
"${python_bin}" - "${public_terminal}" "${submission}" "${digest}" "${member_count}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
public,submission=map(Path,sys.argv[1:3]); digest=sys.argv[3]; members=int(sys.argv[4]); output=Path(sys.argv[5])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"SAVED","public_terminal":str(public.resolve()),
 "submission":str(submission.resolve()),"sha256":digest,"members":members,
 "flat_archive":True,"crc_passed":True,"public_read":False,"hidden_labels_read":False,
 "submission_created":True,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
