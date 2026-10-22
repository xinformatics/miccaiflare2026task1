#!/usr/bin/env bash
set -euo pipefail

# Save-only consumer for the independent FreeTumor proposal source. The exact
# direct-union configuration is selected upstream on the fixed public 50. This
# script either stops before hidden access or reproduces that configuration on
# 100 hidden images and creates one audited archive. It cannot upload.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
free_root="${project_root}/external/FreeTumor"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${free_root}/public_sweeps/independent_baseline_20260825"
public_terminal="${public}/TERMINAL.json"
analysis="${public}/complementarity.json"
checkpoint="${free_root}/pretrained_models/baseline/model_baseline_segmentor.pt"
checkpoint_sha="c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
hidden_images="${project_root}/validation/Validation-Hidden-Images"
anchor_archive="${project_root}/submissions/gf_touchcomp_0825.zip"
anchor_sha="d2521c5493169d09719c3c0ecc1b998051dfe5737e2223f85e10d97f39777d11"
output="${free_root}/public_sweeps/independent_baseline_saved_20260825"
hidden="${free_root}/hidden_predictions/independent_baseline_20260825"
terminal="${output}/TERMINAL.json"
hidden_started="${output}/HIDDEN_INFERENCE_STARTED.json"
lock="${output}/controller.lock"
idle_memory_mib=4096

mkdir -p "${output}" "${hidden}" "${project_root}/submissions"
exec 9>"${lock}"
flock -n 9 || { echo "FreeTumor save-only controller already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" configuration="${4:-}" archive="${5:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${configuration}" \
    "${public_terminal}" "${analysis}" "${checkpoint}" "${hidden_started}" \
    "${archive}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,configuration=sys.argv[1:5]
public,analysis,checkpoint,hidden_started=map(Path,sys.argv[5:9])
archive=Path(sys.argv[9]) if sys.argv[9] else None;output=Path(sys.argv[10])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
submission=None
if archive is not None:
    if not archive.is_file() or len(archive.name)>=64: raise SystemExit("invalid archive identity")
    with zipfile.ZipFile(archive) as handle:
        members=[item for item in handle.infolist() if not item.is_dir()]
        if len(members)!=100 or len({item.filename for item in members})!=100:
            raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members):
            raise SystemExit("archive is not exact flat nonempty NIfTI")
        if handle.testzip() is not None: raise SystemExit("archive CRC failure")
    submission={"path":str(archive.resolve()),"sha256":sha(archive),"members":100}
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "selected_configuration":configuration or None,
 "public_terminal":asset(public),"complementarity":asset(analysis),
 "checkpoint":asset(checkpoint),"hidden_inference_started":asset(hidden_started),
 "submission":submission,"public_read":public.is_file(),
 "hidden_read":hidden_started.is_file() or archive is not None,
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
    "FreeTumor saver failed at line ${line}, status ${status}"
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
      mv --no-clobber "${terminal}" \
        "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected FreeTumor saver terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${public_terminal}" ]]; then
    printf '%s waiting for FreeTumor fixed-public terminal\n' "$(date -Ins)";sleep 30;continue
  fi
  public_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${public_terminal}")
  case "${public_status}" in
    COMPLETE) break ;;
    FAILED) printf '%s waiting for replacement of failed FreeTumor public terminal\n' "$(date -Ins)";sleep 30 ;;
    *) echo "unexpected FreeTumor public status: ${public_status}" >&2;exit 2 ;;
  esac
done

read -r gate configuration minimum dsc nsd < <("${python_bin}" - \
  "${public_terminal}" "${analysis}" "${checkpoint}" "${checkpoint_sha}" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
terminal_path,analysis_path,checkpoint=map(Path,sys.argv[1:4]);expected=sys.argv[4]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
terminal=json.load(open(terminal_path,encoding="utf-8"));analysis=json.load(open(analysis_path,encoding="utf-8"))
if terminal.get("status")!="COMPLETE" or terminal.get("decision")!="PUBLIC_DIAGNOSTIC_ONLY":
    raise SystemExit("FreeTumor public terminal did not complete")
for key,path in (("checkpoint",checkpoint),("complementarity",analysis_path)):
    item=terminal.get(key,{})
    if Path(item.get("path","")).resolve()!=path.resolve() or item.get("sha256")!=sha(path):
        raise SystemExit(f"FreeTumor {key} identity mismatch")
if sha(checkpoint)!=expected: raise SystemExit("FreeTumor checkpoint identity mismatch")
if (analysis.get("status")!="COMPLETE" or analysis.get("case_count")!=50
    or analysis.get("role")!="public_label_diagnostic_only_not_submission_authority"
    or not analysis.get("public_labels_read") or analysis.get("hidden_read")
    or analysis.get("submission_created") or analysis.get("uploaded")):
    raise SystemExit("FreeTumor complementarity provenance mismatch")
anchor=analysis.get("metrics",{}).get("anchor",{})
if abs(float(anchor.get("Lesion_DSC",-1))-0.6004316379500749)>1e-12 or abs(float(anchor.get("Lesion_NSD",-1))-0.5104225107769019)>1e-12:
    raise SystemExit("FreeTumor public anchor metric mismatch")
best=analysis.get("best_direct_union",{});configuration=best.get("configuration","")
match=re.fullmatch(r"union_mm(0|500|1000|3000)",configuration)
if not match: raise SystemExit("invalid FreeTumor direct-union selection")
dsc=float(best.get("Lesion_DSC",-1));nsd=float(best.get("Lesion_NSD",-1))
gate=(analysis.get("direct_union_public_gate") is True and dsc>=0.6154316379500749 and nsd>=0.5104225107769019)
terminal_best=terminal.get("complementarity",{}).get("best_direct_union",{})
if terminal_best!=best or bool(terminal.get("complementarity",{}).get("direct_union_public_gate"))!=gate:
    raise SystemExit("FreeTumor terminal/analysis selection mismatch")
print("PASS" if gate else "FAIL",configuration,int(match.group(1)),dsc,nsd)
PY
)
if [[ "${gate}" != PASS ]]; then
  write_terminal SKIPPED KEEP_TOUCHCOMP \
    "independent FreeTumor direct union missed the fixed-public promotion gate" \
    "${configuration}"
  trap - EXIT
  exit 0
fi

[[ "$(sha256sum "${anchor_archive}" | awk '{print $1}')" == "${anchor_sha}" ]]
[[ "$(sha256sum "${checkpoint}" | awk '{print $1}')" == "${checkpoint_sha}" ]]
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
"${python_bin}" - "${configuration}" "${minimum}" "${dsc}" "${nsd}" \
  "${checkpoint}" "${hidden_started}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
configuration,minimum,dsc,nsd=sys.argv[1:5];checkpoint,output=map(Path,sys.argv[5:])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selected_configuration":configuration,"minimum_component_mm3":int(minimum),
 "fixed_public_DSC":float(dsc),"fixed_public_NSD":float(nsd),
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
 "hidden_labels_read":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le "${idle_memory_mib}" && "${used1}" -le "${idle_memory_mib}" ]]; then break;fi
  printf '%s waiting for FreeTumor hidden GPUs: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

mkdir -p "${hidden}/raw" "${hidden}/anchor" "${hidden}/filtered_mm${minimum}" "${hidden}/union_mm${minimum}"
for shard in 0 1; do
  "${python_bin}" -u "${free_root}/evaluate_public_safe.py" \
    --images "${hidden_images}" --output "${hidden}/raw" --checkpoint "${checkpoint}" \
    --gpu "${shard}" --shard-index "${shard}" --num-shards 2 --workers 2 \
    --sw-batch-size 4 --overlap 0.75 --dataset-role hidden \
    --terminal "${hidden}/SHARD_${shard}.json" > "${hidden}/shard_${shard}.log" 2>&1 &
  shard_pids[${shard}]=$!
done
status=0
wait "${shard_pids[0]}" || status=$?
wait "${shard_pids[1]}" || status=$?
[[ "${status}" -eq 0 ]]

"${python_bin}" - "${hidden_images}" "${hidden}/raw" "${checkpoint_sha}" \
  "${hidden}/SHARD_0.json" "${hidden}/SHARD_1.json" <<'PY'
import json,sys
from pathlib import Path
images,predictions=map(Path,sys.argv[1:3]);expected=sys.argv[3];terminals=list(map(Path,sys.argv[4:]))
names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")}
if len(names)!=100 or {p.name for p in predictions.glob("*.nii.gz") }!=names: raise SystemExit("hidden inventory mismatch")
cases=set()
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("dataset_role")!="hidden"
        or x.get("shard_index")!=index or x.get("num_shards")!=2
        or x.get("checkpoint",{}).get("sha256")!=expected
        or x.get("public_images_read") or x.get("public_labels_read")
        or not x.get("hidden_read") or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(f"invalid hidden shard terminal {index}")
    cases.update(x.get("cases",[]))
if cases!=names: raise SystemExit("hidden shard case coverage mismatch")
PY

unzip -oq "${anchor_archive}" -d "${hidden}/anchor"
[[ $(find "${hidden}/anchor" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
"${python_bin}" -u "${gf_root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden}/raw" --fusion a --min-mm3 "${minimum}" --workers 4 \
  --output "${hidden}/filtered_mm${minimum}" > "${hidden}/filter.log" 2>&1
"${python_bin}" -u "${gf_root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden}/anchor" --pred-b "${hidden}/filtered_mm${minimum}" \
  --fusion union --min-mm3 0 --workers 4 --output "${hidden}/union_mm${minimum}" \
  > "${hidden}/union.log" 2>&1
[[ $(find "${hidden}/union_mm${minimum}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

archive="${project_root}/submissions/gf_ftu_u${minimum}_0825.zip"
"${python_bin}" -u "${project_root}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden}/union_mm${minimum}" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" > "${archive}.sha256"
printf 'configuration\tpublic_DSC\tpublic_NSD\tsubmission\tsha256\n' > "${output}/saved_submission.tsv"
printf '%s\t%s\t%s\t%s\t%s\n' "${configuration}" "${dsc}" "${nsd}" "${archive}" "${digest}" \
  >> "${output}/saved_submission.tsv"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "public-qualified independent FreeTumor union reproduced label-blind on 100 hidden images" \
  "${configuration}" "${archive}"
trap - EXIT

