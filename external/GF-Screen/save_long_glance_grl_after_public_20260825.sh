#!/usr/bin/env bash
set -euo pipefail

# Save-only consumer for the single hash-bound long-Glance state selected on
# the fixed public 50. It reproduces the selected label-blind deployment on
# 100 hidden images and creates exactly one verified archive. It has no upload
# command and creates no prediction when the public gate fails.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
controller_root="${root}/public_sweeps/gf_glance_grl_long30_20260825"
controller_terminal="${controller_root}/TERMINAL.json"
source="${controller_root}/SOURCE.json"
recipe="${controller_root}/TRAINING_RECIPE.json"
public="${root}/public_sweeps/gf_glance_grl_long30_trajectory_20260825"
public_terminal="${public}/TERMINAL.json"
summary="${public}/summary.tsv"
run_dir="${root}/runs/gf_glance_grl_long30_20260825"
saved="${root}/public_sweeps/gf_glance_grl_long30_saved_20260825"
validation="${saved}/VALIDATED_PUBLIC.json"
terminal="${saved}/TERMINAL.json"
hidden_started="${saved}/HIDDEN_INFERENCE_STARTED.json"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
gfconf_archive="${project}/submissions/gf_conf_cbg41_e283.zip"
gfconf_sha256="d8f5f18e860ffd3d0a1dfed44112f0983258320c14f11bf64ab5c1ffd6227e4a"
lock="${root}/runs/save_long_glance_grl_after_public_20260825.lock"

mkdir -p "${saved}" "${project}/submissions" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long Glance saver already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" archive="${4:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${archive}" \
    "${validation}" "${hidden_started}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,archive_text=sys.argv[1:5]
validation,hidden_started,output=map(Path,sys.argv[5:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
archive=Path(archive_text) if archive_text else None
archive_asset=None
if archive is not None:
    if not archive.is_file() or len(archive.name)>=64: raise SystemExit("archive identity invalid")
    with zipfile.ZipFile(archive) as handle:
        members=[item for item in handle.infolist() if not item.is_dir()]
        if len(members)!=100 or len({item.filename for item in members})!=100:
            raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members):
            raise SystemExit("archive must contain 100 nonempty flat NIfTI members")
        if handle.testzip() is not None: raise SystemExit("archive CRC failure")
    archive_asset={"path":str(archive.resolve()),"sha256":sha(archive),"members":100}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "validated_public":asset(validation),"hidden_inference_started":asset(hidden_started),
 "submission":archive_asset,"hidden_labels_read":False,
 "submission_created":archive_asset is not None,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED NO_ARCHIVE \
    "long Glance saver failed at line ${line}, status ${status}"
  exit "${status}"
}
failure_line=unknown
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE|SKIPPED) trap - EXIT; exit 0 ;;
    FAILED)
      preserved_terminal="${saved}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved_terminal}"
      printf '%s preserved failed long Glance saver terminal for resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing long Glance saver status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${controller_terminal}" ]]; then
    printf '%s waiting for long Glance public controller terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  controller_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${controller_terminal}")
  case "${controller_status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed long Glance controller terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected long Glance controller status: ${controller_status}" >&2; exit 2 ;;
  esac
done
for file in "${public_terminal}" "${source}" "${recipe}" "${summary}" \
    "${prompt_config}"; do
  [[ -s "${file}" ]] || { echo "missing long-Glance public asset: ${file}" >&2; exit 2; }
done
decision=$("${python_bin}" -u "${root}/validate_long_glance_grl_public_terminal.py" \
  --controller "${controller_terminal}" --public "${public_terminal}" \
  --source "${source}" --recipe "${recipe}" --summary "${summary}" \
  --run-dir "${run_dir}" --trajectory-root "${public}" --output "${validation}")
if [[ "${decision}" != QUALIFY_SUBMISSION_PREPARATION ]]; then
  [[ "${decision}" == KEEP_TOUCHCOMP ]]
  write_terminal SKIPPED KEEP_TOUCHCOMP \
    "long Glance fixed-public trajectory did not pass the submission gate"
  trap - EXIT
  exit 0
fi

read -r epoch glance_threshold variant dsc nsd checkpoint < <("${python_bin}" - "${validation}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if p.get("status")!="PASS" or p.get("decision")!="QUALIFY_SUBMISSION_PREPARATION":
    raise SystemExit("validated public state did not qualify")
print(p["selected_epoch"],p["selected_glance_threshold"],p["selected_variant"],p["selected_fixed_public_DSC"],p["selected_fixed_public_NSD"],p["checkpoint"]["path"])
PY
)
case "${variant}" in
  deployed500) minimum=500 ;;
  deployed1000|touching) minimum=1000 ;;
  *) echo "unexpected selected deployment: ${variant}" >&2; exit 2 ;;
esac
if [[ "${variant}" == touching ]]; then
  [[ -s "${gfconf_archive}" ]] || { echo "missing immutable gf_conf anchor" >&2; exit 2; }
fi

"${python_bin}" - "${validation}" "${hidden_started}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
validation,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "validated_public":{"path":str(validation.resolve()),"sha256":hashlib.sha256(validation.read_bytes()).hexdigest()},
 "hidden_labels_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected exact 100 hidden assets: ${directory}" >&2; exit 2;
  }
done

threshold_tag="g$(printf '%.2f' "${glance_threshold}" | tr . p)"
hidden="${root}/hidden_predictions/gf_glance_grl_e${epoch}_${threshold_tag}_20260825"
mkdir -p "${hidden}/shard0" "${hidden}/shard1" "${hidden}/raw" \
  "${hidden}/filtered_mm${minimum}" "${hidden}/prompted_mm${minimum}" \
  "${hidden}/deployed_mm${minimum}"
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 1536 && "${used1}" -le 1536 ]]; then break; fi
  printf '%s waiting for long-Glance hidden GPUs: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold "${glance_threshold}" --lesion_threshold 0.50 \
    --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${hidden}/shard${shard}.log" 2>&1
}
infer_shard 0 0 & pid0=$!
infer_shard 1 1 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?; [[ "${status}" -eq 0 ]]
cp "${hidden}/shard0/"*.nii.gz "${hidden}/raw/"
cp "${hidden}/shard1/"*.nii.gz "${hidden}/raw/"
[[ $(find "${hidden}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

filtered="${hidden}/filtered_mm${minimum}"
prompted="${hidden}/prompted_mm${minimum}"
deployed="${hidden}/deployed_mm${minimum}"
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden}/raw" --fusion a --min-mm3 "${minimum}" --workers 4 \
  --output "${filtered}" > "${hidden}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${filtered}" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${prompted}" > "${hidden}/prompt.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${prompted}" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${deployed}" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${hidden}/guided.log" 2>&1
selected="${deployed}"
if [[ "${variant}" == touching ]]; then
  [[ $(sha256sum "${gfconf_archive}" | awk '{print $1}') == "${gfconf_sha256}" ]] || {
    echo "immutable gf_conf archive identity mismatch" >&2; exit 2;
  }
  mkdir -p "${hidden}/gfconf_anchor" "${hidden}/touching"
  unzip -oq "${gfconf_archive}" -d "${hidden}/gfconf_anchor"
  [[ $(find "${hidden}/gfconf_anchor" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
  "${python_bin}" -u "${root}/screen_anchor_support_contour_growth.py" materialize \
    --anchor "${hidden}/gfconf_anchor" --support "${deployed}" \
    --configuration touching_component --output "${hidden}/touching" --workers 4 \
    > "${hidden}/touching.log" 2>&1
  selected="${hidden}/touching"
fi
[[ $(find "${selected}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

case "${variant}" in
  deployed500) suffix=d500 ;;
  deployed1000) suffix=d1k ;;
  touching) suffix=touch ;;
esac
tag="gf_grl${epoch}_${threshold_tag}_${suffix}_0825"
archive="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${selected}" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" > "${archive}.sha256"
printf 'tag\tepoch\tglance_threshold\tvariant\tcheckpoint\tDSC\tNSD\tsubmission\tsha256\n' \
  > "${saved}/saved_submission.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${epoch}" "${glance_threshold}" "${variant}" "${checkpoint}" \
  "${dsc}" "${nsd}" "${archive}" "${digest}" >> "${saved}/saved_submission.tsv"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "public-selected long-Glance deployment reproduced label-blind on 100 hidden images" \
  "${archive}"
trap - EXIT
cat "${saved}/saved_submission.tsv"
