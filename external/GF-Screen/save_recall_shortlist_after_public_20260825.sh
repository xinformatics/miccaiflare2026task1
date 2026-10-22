#!/usr/bin/env bash
set -euo pipefail

# Save-only consumer for a recall-shortlisted checkpoint that independently
# passed the immutable fixed-public gate. Hidden labels are never referenced;
# the resulting archive is for manual upload only.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${root}/public_sweeps/gf_focus_pu_recall_shortlist_public_20260825"
public_terminal="${public}/TERMINAL.json"
summary="${public}/summary.tsv"
shortlist="${root}/public_sweeps/gf_focus_pu_long30_internal_grid_20260825/fixed_public_shortlist.json"
run_dir="${root}/runs/gf_focus_pu_long30_e283_to313_20260825"
saved="${root}/public_sweeps/gf_focus_pu_recall_shortlist_saved_20260825"
contract="${saved}/SAVE_CONTRACT.json"
terminal="${saved}/TERMINAL.json"
hidden_started="${saved}/HIDDEN_INFERENCE_STARTED.json"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
gfconf_archive="${project}/submissions/gf_conf_cbg41_e283.zip"
gfconf_sha256="d8f5f18e860ffd3d0a1dfed44112f0983258320c14f11bf64ab5c1ffd6227e4a"
lock="${root}/runs/save_recall_shortlist_after_public_20260825.lock"
epoch=""
variant=""
checkpoint=""

mkdir -p "${saved}" "${project}/submissions" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "recall-shortlist saver already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" archive="${4:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${epoch}" \
    "${variant}" "${contract}" "${public_terminal}" "${checkpoint}" \
    "${archive}" "${hidden_started}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,epoch,variant=sys.argv[1:6]
contract,public=map(Path,sys.argv[6:8]);checkpoint=Path(sys.argv[8]) if sys.argv[8] else None
archive=Path(sys.argv[9]) if sys.argv[9] else None;hidden_started=Path(sys.argv[10]);output=Path(sys.argv[11])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path is not None and path.is_file() else None
archive_asset=None
if archive is not None:
    if not archive.is_file() or len(archive.name)>=64: raise SystemExit("archive identity invalid")
    with zipfile.ZipFile(archive) as handle:
        members=[item for item in handle.infolist() if not item.is_dir()]
        if len(members)!=100 or len({item.filename for item in members})!=100:
            raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members):
            raise SystemExit("archive is not exact flat nonempty NIfTI")
        if handle.testzip() is not None: raise SystemExit("archive CRC failure")
    archive_asset={"path":str(archive.resolve()),"sha256":sha(archive),"members":100}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "selected_epoch":int(epoch) if epoch else None,"selected_variant":variant or None,
 "save_contract":asset(contract),"public_terminal":asset(public),
 "checkpoint":asset(checkpoint),"submission":archive_asset,
 "hidden_inference_started":asset(hidden_started),
 "public_read":public.is_file(),"hidden_read":hidden_started.is_file() or archive is not None,
 "hidden_labels_read":False,"submission_created":archive is not None,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

failure_line=unknown
failure() {
  local status=$? line="${failure_line}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED NO_ARCHIVE \
    "recall-shortlist saver failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE|SKIPPED) trap - EXIT; exit 0 ;;
    FAILED)
      preserved="${saved}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      mv --no-clobber "${terminal}" "${preserved}"
      ;;
    *) echo "unexpected recall-shortlist saver status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${public_terminal}" ]]; then
    printf '%s waiting for recall-shortlist public terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  public_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${public_terminal}")
  case "${public_status}" in
    COMPLETE) break ;;
    FAILED)
      printf '%s waiting for replacement of failed recall-public terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected recall-public terminal status: ${public_status}" >&2; exit 2 ;;
  esac
done

for file in "${shortlist}" "${summary}" "${prompt_config}" "${gfconf_archive}"; do
  [[ -s "${file}" ]] || { echo "missing recall-save asset: ${file}" >&2; exit 2; }
done
"${python_bin}" -u "${root}/validate_recall_shortlist_save_contract.py" \
  --public-terminal "${public_terminal}" --shortlist "${shortlist}" \
  --summary "${summary}" --run-dir "${run_dir}" --children-root "${public}" \
  --output "${contract}" > "${saved}/contract.log"

contract_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
  "${contract}")
if [[ "${contract_status}" == SKIP ]]; then
  write_terminal SKIPPED KEEP_TOUCHCOMP \
    "no recall-shortlisted checkpoint passed the immutable fixed-public gate"
  trap - EXIT
  exit 0
fi
[[ "${contract_status}" == PASS ]]
read -r epoch variant minimum dsc nsd checkpoint < <("${python_bin}" - "${contract}" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]);payload=json.load(open(path,encoding="utf-8"));identity=payload.get("checkpoint",{});checkpoint=Path(identity.get("path",""))
if payload.get("status")!="PASS" or payload.get("decision")!="GENERATE_HIDDEN_ARCHIVE": raise SystemExit("save contract did not pass")
if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=identity.get("sha256"): raise SystemExit("save checkpoint identity mismatch")
print(payload["selected_epoch"],payload["selected_variant"],payload["minimum_component_mm3"],payload["selected_fixed_public_DSC"],payload["selected_fixed_public_NSD"],checkpoint.resolve())
PY
)
hidden="${root}/hidden_predictions/gf_focus_recall_e${epoch}_20260825"

# Persist the provenance transition before inspecting hidden images/supports.
"${python_bin}" - "${variant}" "${checkpoint}" "${contract}" "${hidden_started}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
variant=sys.argv[1];checkpoint,contract,output=map(Path,sys.argv[2:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selected_variant":variant,"checkpoint":{"path":str(checkpoint.resolve()),"sha256":sha(checkpoint)},
 "save_contract":{"path":str(contract.resolve()),"sha256":sha(contract)},"hidden_labels_read":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected exact 100 hidden assets: ${directory}" >&2; exit 2;
  }
done

mkdir -p "${hidden}/shard0" "${hidden}/shard1" "${hidden}/raw" \
  "${hidden}/filtered_mm${minimum}" "${hidden}/prompted_mm${minimum}" \
  "${hidden}/deployed_mm${minimum}"
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1); free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1536 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1536 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for recall-shortlist hidden GPU: gpu0=%s gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.20 --lesion_threshold 0.50 \
    --focus_batch_size 16 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${hidden}/shard${shard}.log" 2>&1
}
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_shard 0 0 & pid0=$!
  infer_shard 1 1 & pid1=$!
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]]
elif [[ "${free0}" -eq 1 ]]; then
  infer_shard 0 0
  infer_shard 0 1
else
  infer_shard 1 0
  infer_shard 1 1
fi
cp "${hidden}/shard0/"*.nii.gz "${hidden}/raw/"
cp "${hidden}/shard1/"*.nii.gz "${hidden}/raw/"
[[ $(find "${hidden}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

filtered="${hidden}/filtered_mm${minimum}"
prompted="${hidden}/prompted_mm${minimum}"
deployed="${hidden}/deployed_mm${minimum}"
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden}/raw" --fusion a --min-mm3 "${minimum}" --workers 4 \
  --output "${filtered}" > "${hidden}/filter_mm${minimum}.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${filtered}" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${prompted}" \
  > "${hidden}/prompt_mm${minimum}.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${prompted}" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${deployed}" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${hidden}/guided_mm${minimum}.log" 2>&1
selected="${deployed}"
if [[ "${variant}" == touching ]]; then
  [[ $(sha256sum "${gfconf_archive}" | awk '{print $1}') == "${gfconf_sha256}" ]] || {
    echo "immutable gf_conf archive identity mismatch" >&2; exit 2;
  }
  gfconf="${hidden}/gfconf_anchor"
  mkdir -p "${gfconf}" "${hidden}/touching"
  if [[ $(find "${gfconf}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
    unzip -nq "${gfconf_archive}" -d "${gfconf}"
  fi
  "${python_bin}" -u "${root}/screen_anchor_support_contour_growth.py" materialize \
    --anchor "${gfconf}" --support "${deployed}" \
    --configuration touching_component --output "${hidden}/touching" --workers 4 \
    > "${hidden}/touching.log" 2>&1
  selected="${hidden}/touching"
fi
[[ $(find "${selected}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

case "${variant}" in
  deployed500) tag="gf_rcl${epoch}_d500_0825" ;;
  deployed1000) tag="gf_rcl${epoch}_d1k_0825" ;;
  touching) tag="gf_rcl${epoch}_touch_0825" ;;
  *) exit 2 ;;
esac
archive="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${selected}" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" > "${archive}.sha256"
printf 'tag\tvariant\tcheckpoint\tDSC\tNSD\tsubmission\tsha256\n' \
  > "${saved}/saved_submission.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${variant}" \
  "${checkpoint}" "${dsc}" "${nsd}" "${archive}" "${digest}" \
  >> "${saved}/saved_submission.tsv"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "public-qualified recall-shortlisted deployment reproduced label-blind on 100 hidden images" \
  "${archive}"
trap - EXIT
cat "${saved}/saved_submission.tsv"
