#!/usr/bin/env bash
set -euo pipefail

# Save exactly one label-blind hidden-validation archive for the best
# independently qualified candidate across the base, positive-unlabeled,
# organ-target, and dual-window ResEnc families. This script has no upload
# capability and does not compare unqualified rows.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
base_public="${project_root}/nnunet/public_sweeps/resenc_s6_partialbox_endpoints_20260827/TERMINAL.json"
positive_public="${project_root}/nnunet/public_sweeps/resenc_s6_positive_unlabeled_endpoints_20260827/TERMINAL.json"
organ_public="${project_root}/nnunet/public_sweeps/resenc_s6_organ_target_endpoints_20260827/TERMINAL.json"
dualwindow_public="${project_root}/nnunet/public_sweeps/resenc_s6_dualwindow_endpoints_20260827/TERMINAL.json"
public_terminals=("${base_public}" "${positive_public}" "${organ_public}" "${dualwindow_public}")
save_preflight="${project_root}/nnunet/RESENC_S6_DUALWINDOW_SAVE_PREFLIGHT_20260827.json"
validator="${project_root}/nnunet/validate_resenc_s6_global_selection.py"
materializer="${project_root}/nnunet/materialize_resenc_s6_deployment.py"
packager="${project_root}/nnunet/package_codabench.py"
hidden_images="${project_root}/validation/Validation-Hidden-Images"
anchor_archive="${project_root}/submissions/gf_touchcomp_0825.zip"
anchor_sha="d2521c5493169d09719c3c0ecc1b998051dfe5737e2223f85e10d97f39777d11"
freetumor_root="${project_root}/external/FreeTumor/hidden_predictions/independent_baseline_20260825"
freetumor="${freetumor_root}/raw"
freetumor_checkpoint_sha="c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
output="${project_root}/nnunet/public_sweeps/resenc_s6_dualwindow_saved_20260827"
hidden="${project_root}/nnunet/hidden_predictions/resenc_s6_dualwindow_20260827"
terminal="${output}/TERMINAL.json"
hidden_started="${output}/HIDDEN_INFERENCE_STARTED.json"
minimum_disk_available_bytes=$((128 * 1024 * 1024 * 1024))

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${output}/.matplotlib"

mkdir -p "${output}" "${hidden}" "${project_root}/submissions"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "dual-window hidden saver already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" selected_json="${4:-null}" archive="${5:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${selected_json}" \
    "${project_root}" "${save_preflight}" "${hidden_started}" "${archive}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,selected_text=sys.argv[1:5]
root,preflight,started=map(Path,sys.argv[5:8]);archive=Path(sys.argv[8]) if sys.argv[8] else None;output=Path(sys.argv[9])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
asset=lambda path:{"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
public_paths={
 "base":root/"nnunet/public_sweeps/resenc_s6_partialbox_endpoints_20260827/TERMINAL.json",
 "positive":root/"nnunet/public_sweeps/resenc_s6_positive_unlabeled_endpoints_20260827/TERMINAL.json",
 "organ":root/"nnunet/public_sweeps/resenc_s6_organ_target_endpoints_20260827/TERMINAL.json",
 "dualwindow":root/"nnunet/public_sweeps/resenc_s6_dualwindow_endpoints_20260827/TERMINAL.json",
}
public_assets={name:asset(path) for name,path in public_paths.items()}
selected=json.loads(selected_text)
submission=None
if archive is not None:
    if not archive.is_file() or len(archive.name)>=64: raise SystemExit("invalid archive identity")
    with zipfile.ZipFile(archive) as handle:
        members=[item for item in handle.infolist() if not item.is_dir()]
        if len(members)!=100 or len({item.filename for item in members})!=100: raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members):
            raise SystemExit("archive is not exact flat nonempty NIfTI")
        if handle.testzip() is not None: raise SystemExit("archive CRC failure")
    submission={"path":str(archive.resolve()),"sha256":sha(archive),"members":100}
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,"selected":selected,
 "public_terminals":public_assets,"save_preflight":asset(preflight),
 "hidden_inference_started":asset(started),"submission":submission,
 "public_read":any(value is not None for value in public_assets.values()),
 "hidden_read":started.is_file() or archive is not None,
 "hidden_labels_read":False,"submission_created":archive is not None,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY
}

failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED NO_ARCHIVE \
      "global ResEnc hidden saver failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE|SKIPPED) trap - EXIT; exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected global ResEnc saver status: ${existing}" >&2; exit 2 ;;
  esac
fi

for file in "${save_preflight}" "${validator}" "${materializer}" \
  "${packager}" "${anchor_archive}"; do
  [[ -s "${file}" ]] || { echo "missing global ResEnc saver input: ${file}" >&2; exit 2; }
done
[[ "$(sha256sum "${anchor_archive}" | awk '{print $1}')" == "${anchor_sha}" ]]
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" - "${project_root}" "${save_preflight}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,record=map(Path,sys.argv[1:]);payload=json.load(open(record,encoding="utf-8"))
if (payload.get("status")!="PASS" or payload.get("tests",{}).get("total_passed")!=24
    or payload.get("tests",{}).get("total_failed")!=0
    or payload.get("public_read") or payload.get("hidden_read")):
    raise SystemExit("invalid global ResEnc save preflight")
for item in payload.get("code",{}).values():
    path=root/item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
        raise SystemExit(f"global ResEnc saver code hash mismatch: {path}")
PY

for public_terminal in "${public_terminals[@]}"; do
  while :; do
    if [[ ! -s "${public_terminal}" ]]; then
      printf '%s waiting for complete ResEnc family selector: %s\n' \
        "$(date -Ins)" "${public_terminal}"
      sleep 30
      continue
    fi
    status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${public_terminal}")
    case "${status}" in
      COMPLETE) break ;;
      FAILED) printf '%s waiting for replacement ResEnc family terminal: %s\n' \
        "$(date -Ins)" "${public_terminal}"; sleep 30 ;;
      *) echo "unexpected ResEnc family selector status: ${status}" >&2; exit 2 ;;
    esac
  done
done

selected_json=$("${python_bin}" "${validator}" --project-root "${project_root}")
if [[ "${selected_json}" == null ]]; then
  write_terminal SKIPPED NO_HIDDEN_ARCHIVE \
    "no ResEnc family endpoint/deployment passed every frozen public gate" null
  trap - EXIT
  exit 0
fi

read -r family trainer predictor model_dir epoch deployment checkpoint checkpoint_sha dsc nsd < <("${python_bin}" - "${selected_json}" <<'PY'
import json,sys
x=json.loads(sys.argv[1]);print(x["family"],x["trainer"],x["predictor"],x["model_dir"],x["epoch"],x["deployment"],x["checkpoint"],x["checkpoint_sha256"],x["fixed_public_DSC"],x["fixed_public_NSD"])
PY
)
checkpoint_name=$(basename "${checkpoint}")
"${python_bin}" - "${selected_json}" "${hidden_started}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
selected=json.loads(sys.argv[1]);output=Path(sys.argv[2])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"STARTED","selected":selected,"dataset_role":"hidden",
 "hidden_labels_read":False,"submission_created":False,"uploaded":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

while :; do
  read -r available_kib full_avg10 disk_available_bytes < <("${python_bin}" - "${hidden}" <<'PY'
import os,sys
from pathlib import Path
memory={line.split(":",1)[0]:line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
pressure=next(line for line in Path("/proc/pressure/memory").read_text().splitlines() if line.startswith("full "))
avg10=float(next(item.split("=",1)[1] for item in pressure.split() if item.startswith("avg10=")))
filesystem=os.statvfs(sys.argv[1])
disk_available=filesystem.f_bavail*filesystem.f_frsize
print(memory["MemAvailable"],avg10,disk_available)
PY
  )
  used=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  util=$(nvidia-smi --id=1 --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
  if "${python_bin}" - "${available_kib}" "${full_avg10}" "${used}" "${util}" \
    "${disk_available_bytes}" "${minimum_disk_available_bytes}" <<'PY'
import sys
available,pressure,used,util,disk,minimum=(
    int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),
    int(sys.argv[5]),int(sys.argv[6])
)
raise SystemExit(0 if (available>=30*1024*1024 and pressure<=2.0
    and used<=3072 and util<=5 and disk>=minimum) else 1)
PY
  then break; fi
  printf '%s waiting for safe dual-window hidden inference: available=%s pressure=%s gpu1=%s/%s%% disk=%s minimum_disk=%s bytes\n' \
    "$(date -Ins)" "${available_kib}" "${full_avg10}" "${used}" "${util}" \
    "${disk_available_bytes}" "${minimum_disk_available_bytes}"
  sleep 30
done

mkdir -p "${hidden}/resenc" "${hidden}/anchor" "${hidden}/deployment"
for part_id in $(seq 0 99); do
  part4=$(printf '%04d' "${part_id}")
  case_terminal="${hidden}/RESENC_CASE_${part4}.json"
  valid=0
  if [[ -s "${case_terminal}" ]] && "${python_bin}" - \
    "${case_terminal}" "${hidden}/resenc" "${checkpoint_sha}" "${part_id}" "${trainer}" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8"));pred=Path(sys.argv[2]);sha=sys.argv[3];part=int(sys.argv[4]);trainer=sys.argv[5];cases=p.get("cases") or []
if (p.get("status")!="COMPLETE" or p.get("trainer")!=trainer
    or p.get("dataset_role")!="hidden" or p.get("part_id")!=part or p.get("num_parts")!=100
    or p.get("case_count")!=1 or len(cases)!=1 or p.get("checkpoint",{}).get("sha256")!=sha
    or not p.get("direct_compact_probabilities") or p.get("public_images_read")
    or p.get("public_labels_read") or not p.get("hidden_read") or p.get("hidden_labels_read")
    or p.get("submission_created") or p.get("uploaded")):
    raise SystemExit(1)
name=cases[0]
if not (pred/name).is_file() or not (pred/f"{name.removesuffix('.nii.gz')}.prob16.npz").is_file(): raise SystemExit(1)
PY
  then valid=1; fi
  [[ "${valid}" -eq 1 ]] && continue
  "${python_bin}" -u "${predictor}" --images "${hidden_images}" \
    --output "${hidden}/resenc" --model-dir "${model_dir}" \
    --checkpoint "${checkpoint_name}" --gpu 1 --part-id "${part_id}" --num-parts 100 \
    --workers 1 --save-probabilities --direct-compact-probabilities --sequential \
    --dataset-role hidden --terminal "${case_terminal}" \
    >"${hidden}/resenc_case_${part4}.log" 2>&1
done

"${python_bin}" - "${hidden_images}" "${hidden}/resenc" "${hidden}" "${checkpoint_sha}" "${trainer}" <<'PY'
import json,sys
from pathlib import Path
images,predictions,root=map(Path,sys.argv[1:4]);expected_sha=sys.argv[4];trainer=sys.argv[5]
names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")};cases=set();terminals=sorted(root.glob("RESENC_CASE_*.json"))
if len(names)!=100 or {p.name for p in predictions.glob("*.nii.gz")}!=names: raise SystemExit("global ResEnc hidden inventory mismatch")
if {p.name for p in predictions.glob("*.prob16.npz")}!={n.removesuffix(".nii.gz")+".prob16.npz" for n in names}: raise SystemExit("global ResEnc hidden probability mismatch")
if len(terminals)!=100: raise SystemExit("global ResEnc hidden terminal coverage mismatch")
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("trainer")!=trainer
        or x.get("dataset_role")!="hidden" or x.get("part_id")!=index or x.get("num_parts")!=100
        or x.get("checkpoint",{}).get("sha256")!=expected_sha or x.get("public_images_read")
        or x.get("public_labels_read") or not x.get("hidden_read") or x.get("hidden_labels_read")
        or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(f"invalid global ResEnc hidden terminal {index}")
    cases.update(x.get("cases",[]))
if cases!=names: raise SystemExit("dual-window hidden case coverage mismatch")
PY

"${python_bin}" - "${hidden_images}" "${freetumor}" "${freetumor_root}" "${freetumor_checkpoint_sha}" <<'PY'
import json,sys
from pathlib import Path
images,predictions,root=map(Path,sys.argv[1:4]);expected_sha=sys.argv[4]
names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")};terminals=sorted(root.glob("FREE_CASE_*.json"));cases=set()
if len(names)!=100 or {p.name for p in predictions.glob("*.nii.gz")}!=names or len(terminals)!=100: raise SystemExit("FreeTumor hidden assets incomplete")
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("dataset_role")!="hidden"
        or x.get("shard_index")!=index or x.get("num_shards")!=100 or x.get("case_count")!=1
        or x.get("checkpoint",{}).get("sha256")!=expected_sha or x.get("public_images_read")
        or x.get("public_labels_read") or not x.get("hidden_read")
        or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(f"invalid FreeTumor hidden terminal {index}")
    cases.update(x.get("cases",[]))
if cases!=names: raise SystemExit("FreeTumor hidden case coverage mismatch")
PY

unzip -oq "${anchor_archive}" -d "${hidden}/anchor"
[[ $(find "${hidden}/anchor" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
"${python_bin}" -u "${materializer}" --deployment "${deployment}" \
  --anchor "${hidden}/anchor" --resenc "${hidden}/resenc" --freetumor "${freetumor}" \
  --output "${hidden}/deployment" --terminal "${hidden}/MATERIALIZE_TERMINAL.json" \
  --expected-count 100 --workers 2 --dataset-role hidden \
  >"${hidden}/materialize.log" 2>&1

case "${deployment}" in
  resenc_standalone_p50) short="p50" ;;
  gf_anchor_confirm_either_p70) short="confp70" ;;
  gf_resenc_boundary3mm_p70p50_ftu) short="b3p70" ;;
  gf_union_resenc_p40_ftu_s10_mm3000) short="u40s10m3k" ;;
  *) echo "unreachable deployment: ${deployment}" >&2; exit 2 ;;
esac
archive="${project_root}/submissions/rs6_${family}_e${epoch}_${short}_0827.zip"
"${python_bin}" -u "${packager}" --images "${hidden_images}" \
  --pred-dir "${hidden}/deployment" --output "${archive}" --expected-count 100 \
  >"${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" >"${archive}.sha256"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "best public-qualified ResEnc family reproduced label-blind on 100 hidden images" \
  "${selected_json}" "${archive}"
trap - EXIT
