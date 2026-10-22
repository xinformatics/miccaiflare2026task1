#!/usr/bin/env bash
set -euo pipefail

# Save-only hidden materialization of the explicitly non-primary TTA evidence
# floor. The original strict TTA rejection remains authoritative. This path
# waits until the higher-priority exact-707 GPU inference cache is complete and
# has no upload or network code. It can then run while OOF threshold scoring
# continues on CPUs, rather than waiting behind several longer training arms.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public_root="${root}/public_sweeps/gf_champion_focus_tta_20260824"
selection="${public_root}/EVIDENCE_FLOOR_SELECTION.json"
terminal="${public_root}/TTA_FALLBACK_SAVE_TERMINAL.json"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
hidden_root="${public_root}/hidden_tta_xy_evidence_floor"
archive="${project}/submissions/gf_ttaxy_fb_0824.zip"
gpu_poll_seconds="${GPU_POLL_SECONDS:-15}"

higher_priority_inference_markers=(
  "${root}/analysis/candidate_refiner_v4_cell707_training_20260823/TRAINING_COMPLETE.json"
  "${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/fold0/work_bbox24/INFERENCE_COMPLETE.json"
  "${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/fold1/work_bbox24/INFERENCE_COMPLETE.json"
  "${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/fold2/work_bbox24/INFERENCE_COMPLETE.json"
  "${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/fold3/work_bbox24/INFERENCE_COMPLETE.json"
  "${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/fold4/work_bbox24/INFERENCE_COMPLETE.json"
)

mkdir -p "${public_root}" "${root}/runs" "${project}/submissions"
exec 9>"${root}/runs/save_focus_tta_evidence_floor_20260824.lock"
flock -n 9 || { echo "TTA evidence-floor saver already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

for required in "${selection}" "${checkpoint}" "${prompt_config}"; do
  [[ -s "${required}" ]] || { echo "missing TTA fallback asset: ${required}" >&2; exit 2; }
done
for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 hidden files: ${directory}" >&2; exit 2;
  }
done

"${python_bin}" - "${selection}" <<'PY'
import hashlib,json,sys
from pathlib import Path
selection=Path(sys.argv[1]); payload=json.loads(selection.read_text(encoding="utf-8"))
if payload.get("decision")!="SAVE_EXPLORATORY_TTA_FALLBACK":
    raise SystemExit("TTA evidence floor did not select a fallback")
if payload.get("selected_variant")!="tta_xy":
    raise SystemExit("unexpected TTA evidence-floor variant")
if payload.get("status_class")!="POST_HOC_EVIDENCE_FLOOR_NOT_PRIMARY_PROMOTION":
    raise SystemExit("TTA evidence-floor status class changed")
if payload.get("automatic_upload_authorized") is not False:
    raise SystemExit("TTA evidence floor must remain save-only")
for item in payload.get("inputs",{}).values():
    path=Path(item["path"])
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
        raise SystemExit(f"TTA evidence input identity mismatch: {path}")
PY

while :; do
  missing=()
  for prerequisite in "${higher_priority_inference_markers[@]}"; do
    [[ -s "${prerequisite}" ]] || missing+=("${prerequisite}")
  done
  [[ "${#missing[@]}" -eq 0 ]] && break
  printf '%s waiting for %s higher-priority training/inference markers\n' \
    "$(date -Ins)" "${#missing[@]}"
  sleep 30
done

[[ "${gpu_poll_seconds}" =~ ^[0-9]+$ && "${gpu_poll_seconds}" -ge 5 ]] || {
  echo "GPU_POLL_SECONDS must be an integer >=5" >&2; exit 2;
}
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1536 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1536 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for TTA fallback GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep "${gpu_poll_seconds}"
done

mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" \
  "${hidden_root}/filtered" "${hidden_root}/prompted" "${hidden_root}/deployed"
infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_tta_axes xy --focus_batch_size 2 --workers 2 \
    --num_shards 2 --shard_id "${shard}" --skip_existing \
    > "${hidden_root}/shard${shard}.log" 2>&1
}
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_shard 0 0 & pid0=$!; infer_shard 1 1 & pid1=$!
  status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  infer_shard 0 0; infer_shard 0 1
else
  infer_shard 1 0; infer_shard 1 1
fi

cp "${hidden_root}/shard0/"*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1/"*.nii.gz "${hidden_root}/raw/"
[[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "TTA fallback hidden inference is incomplete" >&2; exit 3;
}
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden_root}/raw" --fusion a --min-mm3 1000 --workers 4 \
  --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompted" \
  > "${hidden_root}/prompted.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${hidden_root}/prompted" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${hidden_root}/deployed" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${hidden_root}/deployed.log" 2>&1
[[ $(find "${hidden_root}/deployed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "TTA fallback hidden deployment is incomplete" >&2; exit 3;
}

"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/deployed" \
  --output "${archive}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
"${python_bin}" - "${archive}" <<'PY'
import sys,zipfile
from pathlib import Path
path=Path(sys.argv[1])
if len(path.name)>=64:raise SystemExit("archive basename is too long")
with zipfile.ZipFile(path) as archive:
    files=[item for item in archive.infolist() if not item.is_dir()]
    names=[item.filename for item in files]
    if len(names)!=100 or len(set(names))!=100:
        raise SystemExit("archive is not exact-100")
    if any("/" in name or "\\" in name or not name.endswith(".nii.gz") for name in names):
        raise SystemExit("archive members are not flat NIfTI files")
    if any(item.file_size<1 for item in files) or archive.testzip() is not None:
        raise SystemExit("archive CRC or member-size validation failed")
PY
sha256sum "${archive}" > "${archive}.sha256"

"${python_bin}" - "${selection}" "${checkpoint}" "${archive}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
selection,checkpoint,archive,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"SAVED_EXPLORATORY_TTA_FALLBACK",
 "status_class":"POST_HOC_EVIDENCE_FLOOR_NOT_PRIMARY_PROMOTION","variant":"tta_xy",
 "selection":str(selection.resolve()),"selection_sha256":sha(selection),
 "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha(checkpoint),
 "archive":str(archive.resolve()),"archive_sha256":sha(archive),"members":100,
 "public_read":True,"hidden_read":True,"hidden_labels_read":False,
 "submission_created":True,"automatic_upload_authorized":False,"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
