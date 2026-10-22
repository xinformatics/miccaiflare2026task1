#!/usr/bin/env bash
set -euo pipefail

# Materialize hidden TTA predictions only after the fixed-public final-deployment
# gate promotes them. The output is saved and audited locally; this script has
# no upload or network operation.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public_root="${root}/public_sweeps/gf_champion_focus_tta_20260824"
public_terminal="${public_root}/PUBLIC_TTA_TERMINAL.json"
selection="${public_root}/hidden_selection.json"
terminal="${public_root}/TTA_SAVE_TERMINAL.json"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
gpu_poll_seconds="${GPU_POLL_SECONDS:-10}"

mkdir -p "${public_root}" "${root}/runs" "${project}/submissions"
exec 9>"${root}/runs/save_focus_tta_champion_candidate_20260824.lock"
flock -n 9 || { echo "champion TTA save controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for fixed-public TTA terminal\n' "$(date -Ins)"
  sleep 30
done
public_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${public_terminal}")
if [[ "${public_status}" == COMPLETE ]]; then
  selection_status=0
  "${python_bin}" -u "${root}/select_focus_tta_candidate.py" \
    --summary "${public_root}/summary.tsv" --output "${selection}" \
    > "${public_root}/hidden_selection.log" 2>&1 || selection_status=$?
  if [[ "${selection_status}" -ne 0 ]]; then
    "${python_bin}" - "${public_terminal}" "${selection_status}" "${selection}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source=Path(sys.argv[1]); selector_status=int(sys.argv[2]); output=Path(sys.argv[3])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "decision":"KEEP_CHAMPION","selected":None,
 "reason":"public TTA selector rejected an invalid or non-equivalent baseline",
 "selector_exit_status":selector_status,"public_terminal":str(source.resolve()),
 "public_read":True,"hidden_read":False,"submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  fi
else
  "${python_bin}" - "${public_terminal}" "${selection}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "decision":"KEEP_CHAMPION","selected":None,
 "reason":"public TTA audit did not complete successfully",
 "public_terminal":str(source.resolve()),"public_read":True,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
fi

decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${selection}")
if [[ "${decision}" != PROMOTE_TTA_HIDDEN ]]; then
  "${python_bin}" - "${public_terminal}" "${selection}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
public,selection,output=map(Path,sys.argv[1:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_CHAMPION",
 "public_terminal":str(public.resolve()),"public_terminal_sha256":sha(public),
 "selection":str(selection.resolve()),"selection_sha256":sha(selection),
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi

variant=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["variant"])' \
  "${selection}")
case "${variant}" in
  tta_x) axes=x ;;
  tta_xy) axes=xy ;;
  *) echo "invalid selected TTA variant: ${variant}" >&2; exit 2 ;;
esac

for required in "${checkpoint}" "${prompt_config}"; do
  [[ -s "${required}" ]] || { echo "missing hidden deployment asset: ${required}" >&2; exit 2; }
done
for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected 100 hidden files: ${directory}" >&2; exit 2;
  }
done

hidden_root="${public_root}/hidden_${variant}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" "${hidden_root}/raw" \
  "${hidden_root}/filtered" "${hidden_root}/prompted" "${hidden_root}/deployed"

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
[[ "${gpu_poll_seconds}" =~ ^[0-9]+$ && "${gpu_poll_seconds}" -ge 5 ]] || {
  echo "GPU_POLL_SECONDS must be an integer >=5" >&2
  exit 2
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1536 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1536 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for TTA hidden GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep "${gpu_poll_seconds}"
done

infer_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_tta_axes "${axes}" --focus_batch_size 2 --workers 2 \
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
  echo "TTA hidden inference is incomplete" >&2; exit 3;
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
  echo "TTA hidden deployment is incomplete" >&2; exit 3;
}

archive="${project}/submissions/gf_${variant}_new2k_0824.zip"
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/deployed" \
  --output "${archive}" --expected-count 100 > "${hidden_root}/package.log" 2>&1
"${python_bin}" - "${archive}" <<'PY'
import sys,zipfile
from pathlib import Path
path=Path(sys.argv[1])
if len(path.name)>=64: raise SystemExit("archive basename is too long")
with zipfile.ZipFile(path) as archive:
    names=archive.namelist()
    if len(names)!=100 or len(set(names))!=100 or any("/" in name or not name.endswith(".nii.gz") for name in names):
        raise SystemExit("archive does not contain 100 unique flat NIfTI members")
    if archive.testzip() is not None or any(info.file_size<1 for info in archive.infolist()):
        raise SystemExit("archive CRC or member-size validation failed")
PY
sha256sum "${archive}" > "${archive}.sha256"

"${python_bin}" - "${public_terminal}" "${selection}" "${checkpoint}" \
  "${archive}" "${variant}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
public,selection,checkpoint,archive=map(Path,sys.argv[1:5]); variant=sys.argv[5]; output=Path(sys.argv[6])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"SAVE_TTA_CANDIDATE","variant":variant,
 "public_terminal":str(public.resolve()),"public_terminal_sha256":sha(public),
 "selection":str(selection.resolve()),"selection_sha256":sha(selection),
 "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha(checkpoint),
 "archive":str(archive.resolve()),"archive_sha256":sha(archive),"members":100,
 "public_read":True,"hidden_read":True,"hidden_labels_read":False,
 "submission_created":True,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
