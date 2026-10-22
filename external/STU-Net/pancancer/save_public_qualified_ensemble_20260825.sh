#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# Exact hidden replay of one fixed-public-qualified STU-Net ensemble. Hidden
# labels are unreachable and no upload operation exists.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
stunet_root="${project_root}/external/STU-Net"
free_root="${project_root}/external/FreeTumor"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${STUNET_PUBLIC_OUTPUT:-${stunet_root}/pancancer/public_sweeps/stunetb_ensemble_trajectory_20260825}"
public_terminal="${public}/TERMINAL.json"
summary="${public}/summary.tsv"
model_dir="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
component_model_dir="${STUNET_COMPONENT_MODEL_DIR:-${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox}"
hidden_images="${project_root}/validation/Validation-Hidden-Images"
anchor_archive="${project_root}/submissions/gf_touchcomp_0825.zip"
anchor_sha="d2521c5493169d09719c3c0ecc1b998051dfe5737e2223f85e10d97f39777d11"
free_checkpoint="${free_root}/pretrained_models/baseline/model_baseline_segmentor.pt"
free_checkpoint_sha="c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0"
free_hidden_root="${free_root}/hidden_predictions/independent_baseline_20260825"
free_hidden="${free_hidden_root}/raw"
output="${STUNET_SAVE_OUTPUT:-${stunet_root}/pancancer/public_sweeps/stunetb_ensemble_saved_20260825}"
hidden="${STUNET_HIDDEN_OUTPUT:-${stunet_root}/pancancer/hidden_predictions/stunetb_ensemble_20260825}"
allowed_epochs_csv="${STUNET_ALLOWED_EPOCHS:-100,200,300}"
terminal="${output}/TERMINAL.json"
hidden_started="${output}/HIDDEN_INFERENCE_STARTED.json"
compact_equivalence="${stunet_root}/pancancer/public_sweeps/direct_compact_equivalence_e100_20260826/EQUIVALENCE.json"
lock="${output}/controller.lock"
idle_memory_mib=4096
stunet_hidden_gpu="${STUNET_HIDDEN_GPU:-}"
[[ -z "${stunet_hidden_gpu}" || "${stunet_hidden_gpu}" == 0 || "${stunet_hidden_gpu}" == 1 ]] || {
  echo "STUNET_HIDDEN_GPU must be empty, 0, or 1" >&2
  exit 2
}

mkdir -p "${output}" "${hidden}" "${project_root}/submissions"
exec 9>"${lock}"
flock -n 9 || { echo "STU-Net ensemble saver already active" >&2;exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" configuration="${4:-}" epoch="${5:-}" archive="${6:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${configuration}" "${epoch}" \
    "${public_terminal}" "${summary}" "${hidden_started}" "${archive}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,configuration,epoch=sys.argv[1:6]
public,summary,started=map(Path,sys.argv[6:9]);archive=Path(sys.argv[9]) if sys.argv[9] else None;output=Path(sys.argv[10])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(p): return {"path":str(p.resolve()),"sha256":sha(p)} if p.is_file() else None
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
 "status":status,"decision":decision,"reason":reason,
 "selected_epoch":int(epoch) if epoch else None,"selected_configuration":configuration or None,
 "public_terminal":asset(public),"public_summary":asset(summary),
 "hidden_inference_started":asset(started),"submission":submission,
 "public_read":public.is_file(),"hidden_read":started.is_file() or archive is not None,
 "hidden_labels_read":False,"submission_created":archive is not None,"uploaded":False}
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
    "STU-Net ensemble saver failed at line ${line}, status ${status}"
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap failure EXIT

if [[ -s "${terminal}" ]]; then
  existing=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${existing}" in
    COMPLETE|SKIPPED) trap - EXIT;exit 0 ;;
    FAILED) mv --no-clobber "${terminal}" "${output}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json" ;;
    *) echo "unexpected STU-Net saver status: ${existing}" >&2;exit 2 ;;
  esac
fi
while :; do
  if [[ ! -s "${public_terminal}" ]]; then printf '%s waiting for STU-Net public ensemble terminal\n' "$(date -Ins)";sleep 30;continue;fi
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${public_terminal}")
  case "${status}" in
    COMPLETE) break ;;
    FAILED) printf '%s waiting for replacement of failed STU-Net public terminal\n' "$(date -Ins)";sleep 30 ;;
    *) echo "unexpected STU-Net public status: ${status}" >&2;exit 2 ;;
  esac
done

public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]; then
  [[ "${public_decision}" == KEEP_TOUCHCOMP ]]
  write_terminal SKIPPED KEEP_TOUCHCOMP "STU-Net ensemble missed the fixed-public promotion gate"
  trap - EXIT;exit 0
fi

read -r epoch configuration checkpoint dsc nsd needs_support model_dir < <("${python_bin}" - \
  "${public_terminal}" "${summary}" "${model_dir}" "${component_model_dir}" "${allowed_epochs_csv}" <<'PY'
import csv,hashlib,json,re,sys
from pathlib import Path
public_path,summary_path,base_model_dir,component_model_dir=map(Path,sys.argv[1:5]);allowed=sorted({int(x) for x in sys.argv[5].split(",")});public=json.load(open(public_path,encoding="utf-8"));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (public.get("status")!="COMPLETE" or public.get("decision")!="QUALIFY_SUBMISSION_PREPARATION"
    or public.get("predeclared_epochs")!=allowed or public.get("hidden_read")
    or public.get("submission_created") or public.get("uploaded")):
    raise SystemExit("invalid qualified STU-Net public terminal")
summary_item=public.get("summary",{})
if Path(summary_item.get("path","")).resolve()!=summary_path.resolve() or summary_item.get("sha256")!=sha(summary_path):
    raise SystemExit("STU-Net public summary identity mismatch")
epoch=int(public["selected_epoch"]);configuration=public["selected_configuration"]
branch=public.get("selected_branch","base_global_box")
if branch=="base_global_box": model_dir=base_model_dir
elif branch=="component_box": model_dir=component_model_dir
else: raise SystemExit("invalid selected STU-Net branch")
if epoch not in allowed: raise SystemExit("invalid STU-Net selected epoch")
standalone=configuration=="stunet_standalone"
calibrated=re.fullmatch(r"stunet_p(30|40|60|70)_mm1000",configuration)
calibrated_union=re.fullmatch(r"gf_union_stu_p(30|40|60|70)_mm3000",configuration)
calibrated_supported=re.fullmatch(r"gf_union_stu_p(30|40)_ftu_s10_mm3000",configuration)
filtered=re.fullmatch(r"stunet_mm(500|1000|3000)",configuration)
standalone_supported=re.fullmatch(r"stunet_ftu_s(01|10)_mm(500|1000|3000)",configuration)
direct=re.fullmatch(r"gf_union_stu_mm(500|1000|3000)",configuration)
supported=re.fullmatch(r"gf_union_stu_ftu_s(01|10)_mm(500|1000|3000)",configuration)
confirmed=re.fullmatch(r"gf_anchor_confirm_(stu|ftu|either|both)_any",configuration)
confirmed_probability=re.fullmatch(r"gf_anchor_confirm_either_p(70)",configuration)
if not (standalone or calibrated or calibrated_union or calibrated_supported or filtered or standalone_supported or direct or supported or confirmed or confirmed_probability):
    raise SystemExit("invalid STU-Net selected configuration")
item=public.get("checkpoint",{});checkpoint=Path(item.get("path",""))
allowed_checkpoints={
    (model_dir/"fold_all"/f"checkpoint_epoch_{epoch:04d}.pth").resolve(),
    (model_dir/"fold_all"/f"checkpoint_inference_epoch_{epoch:04d}.pth").resolve(),
}
if checkpoint.resolve() not in allowed_checkpoints or item.get("sha256")!=sha(checkpoint):
    raise SystemExit("STU-Net selected checkpoint identity mismatch")
analysis_path=Path(public["selected_analysis"]["path"]);analysis=json.load(open(analysis_path,encoding="utf-8"))
if public["selected_analysis"]["sha256"]!=sha(analysis_path) or analysis.get("role")!="fixed_public_ensemble_selection_only":
    raise SystemExit("STU-Net selected analysis identity mismatch")
best=analysis.get("best_deployment",{})
if best.get("configuration")!=configuration: raise SystemExit("STU-Net selected deployment mismatch")
dsc=float(public["selected_fixed_public_DSC"]);nsd=float(public["selected_fixed_public_NSD"])
if dsc!=float(best["Lesion_DSC"]) or nsd!=float(best["Lesion_NSD"]) or dsc<0.6154316379500749 or nsd<0.5104225107769019:
    raise SystemExit("STU-Net selected public metrics missed gate")
rows=list(csv.DictReader(open(summary_path,encoding="utf-8"),delimiter="\t"))
matches=[r for r in rows if int(r["checkpoint_epoch"])==epoch and r["configuration"]==configuration
         and r.get("branch","base_global_box")==branch]
if len(matches)!=1 or float(matches[0]["Lesion_DSC"])!=dsc or float(matches[0]["Lesion_NSD"])!=nsd:
    raise SystemExit("STU-Net selected summary row mismatch")
needs_support=(supported or standalone_supported or calibrated_supported or confirmed_probability or (confirmed and confirmed.group(1) in ("ftu","either","both")))
print(epoch,configuration,checkpoint.resolve(),dsc,nsd,1 if needs_support else 0,model_dir.resolve())
PY
)
[[ -s "${checkpoint}" ]]
[[ "$(sha256sum "${anchor_archive}" | awk '{print $1}')" == "${anchor_sha}" ]]
[[ "$(sha256sum "${free_checkpoint}" | awk '{print $1}')" == "${free_checkpoint_sha}" ]]
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
"${python_bin}" - "${epoch}" "${configuration}" "${dsc}" "${nsd}" "${checkpoint}" "${hidden_started}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
epoch,configuration,dsc,nsd=sys.argv[1:5];checkpoint,output=map(Path,sys.argv[5:])
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selected_epoch":int(epoch),"selected_configuration":configuration,
 "fixed_public_DSC":float(dsc),"fixed_public_NSD":float(nsd),
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
 "hidden_labels_read":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(temporary,output)
PY

while :; do
  if [[ -n "${stunet_hidden_gpu}" ]]; then
    used=$(nvidia-smi --id="${stunet_hidden_gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le "${idle_memory_mib}" ]]; then break;fi
    printf '%s waiting for STU-Net hidden GPU %s: %s MiB\n' "$(date -Ins)" "${stunet_hidden_gpu}" "${used}"
  else
    used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le "${idle_memory_mib}" && "${used1}" -le "${idle_memory_mib}" ]]; then break;fi
    printf '%s waiting for STU-Net hidden GPUs: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  fi
  sleep 30
done

mkdir -p "${hidden}/stunet" "${hidden}/anchor" "${hidden}/ensemble"
checkpoint_name=$(basename "${checkpoint}")
compact_args=()
compact_source="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0100.pth"
if [[ -s "${compact_equivalence}" && -s "${compact_source}" ]] && \
  "${python_bin}" - "${compact_equivalence}" "${compact_source}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,source=map(Path,sys.argv[1:]);x=json.load(open(terminal,encoding="utf-8"));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if (x.get("status")!="COMPLETE" or not x.get("activation_authorized")
    or not x.get("mask_equal") or not x.get("probability_uint16_equal") or not x.get("geometry_equal")
    or x.get("maximum_probability_code_difference")!=0
    or x.get("checkpoint",{}).get("sha256")!=sha(source)
    or x.get("public_labels_read") or x.get("hidden_read")
    or x.get("submission_created") or x.get("uploaded")):
    raise SystemExit(1)
PY
then compact_args=(--direct-compact-probabilities);fi
# Probability export is bounded to one hidden CT per process. Long-lived
# multi-case shards and even two concurrent cases crossed this host's managed
# memory-pressure threshold during the epoch-100 public screen.
for part_id in $(seq 0 99); do
  if [[ -n "${stunet_hidden_gpu}" ]]; then gpu="${stunet_hidden_gpu}";else gpu=$((part_id % 2));fi
  part4=$(printf '%04d' "${part_id}")
  "${python_bin}" -u "${stunet_root}/pancancer/predict_stunet_safe.py" \
    --images "${hidden_images}" --output "${hidden}/stunet" --model-dir "${model_dir}" \
    --checkpoint "${checkpoint_name}" --gpu "${gpu}" --part-id "${part_id}" --num-parts 100 \
    --workers 1 --save-probabilities --sequential "${compact_args[@]}" --dataset-role hidden \
    --terminal "${hidden}/STUNET_CASE_${part4}.json" \
    > "${hidden}/stunet_case_${part4}.log" 2>&1
done
[[ $(find "${hidden}/stunet" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${hidden}/stunet" -maxdepth 1 -name '*.prob16.npz' | wc -l) -eq 100 ]]
"${python_bin}" - "${hidden_images}" "${hidden}/stunet" "${checkpoint}" \
  "${hidden}" <<'PY'
import hashlib,json,sys
from pathlib import Path
images,predictions,checkpoint,terminal_root=map(Path,sys.argv[1:]);terminals=sorted(terminal_root.glob("STUNET_CASE_*.json"));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")};cases=set()
if len(names)!=100 or {p.name for p in predictions.glob("*.nii.gz")}!=names: raise SystemExit("STU-Net hidden inventory mismatch")
if len(terminals)!=100: raise SystemExit("STU-Net hidden case-terminal coverage mismatch")
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("dataset_role")!="hidden" or x.get("part_id")!=index
        or x.get("num_parts")!=100 or x.get("case_count")!=1
        or x.get("checkpoint",{}).get("sha256")!=sha(checkpoint)
        or x.get("public_images_read") or x.get("public_labels_read") or not x.get("hidden_read")
        or x.get("hidden_labels_read") or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(f"invalid STU-Net hidden shard terminal {index}")
    cases.update(x.get("cases",[]))
if cases!=names: raise SystemExit("STU-Net hidden shard coverage mismatch")
PY

if [[ "${needs_support}" -eq 1 ]]; then
  mkdir -p "${free_hidden_root}" "${free_hidden}"
  reuse_free=0
  if [[ $(find "${free_hidden}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 \
      && $(find "${free_hidden_root}" -maxdepth 1 -name 'FREE_CASE_*.json' | wc -l) -eq 100 ]]; then
    if "${python_bin}" - "${free_hidden}" "${free_hidden_root}" "${free_checkpoint_sha}" <<'PY'
import json,sys
from pathlib import Path
predictions,root=map(Path,sys.argv[1:3]);expected=sys.argv[3];cases=set()
terminals=sorted(root.glob("FREE_CASE_*.json"))
if len(terminals)!=100: raise SystemExit(1)
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("dataset_role")!="hidden" or x.get("part_id") is not None
        or x.get("shard_index")!=index or x.get("num_shards")!=100 or x.get("case_count")!=1
        or x.get("checkpoint",{}).get("sha256")!=expected
        or x.get("public_images_read") or x.get("public_labels_read") or not x.get("hidden_read")
        or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(1)
    cases.update(x.get("cases",[]))
if len(cases)!=100 or len(list(predictions.glob("*.nii.gz")))!=100: raise SystemExit(1)
PY
    then reuse_free=1;fi
  fi
  if [[ "${reuse_free}" -eq 0 ]]; then
    # FreeTumor is also isolated to one CT per process. Two concurrent
    # long-lived 50-case shards previously crossed systemd-oomd's host-memory
    # threshold even when each GPU had ample VRAM.
    for shard in $(seq 0 99); do
      gpu=$((shard % 2))
      shard4=$(printf '%04d' "${shard}")
      "${python_bin}" -u "${free_root}/evaluate_public_safe.py" \
        --images "${hidden_images}" --output "${free_hidden}" --checkpoint "${free_checkpoint}" \
        --gpu "${gpu}" --shard-index "${shard}" --num-shards 100 --workers 0 \
        --sw-batch-size 1 --overlap 0.75 --dataset-role hidden \
        --terminal "${free_hidden_root}/FREE_CASE_${shard4}.json" \
        > "${free_hidden_root}/free_case_${shard4}.log" 2>&1
    done
  fi
  [[ $(find "${free_hidden}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
  "${python_bin}" - "${hidden_images}" "${free_hidden}" "${free_checkpoint_sha}" \
    "${free_hidden_root}" <<'PY'
import json,sys
from pathlib import Path
images,predictions=map(Path,sys.argv[1:3]);expected=sys.argv[3];root=Path(sys.argv[4]);terminals=sorted(root.glob("FREE_CASE_*.json"));names={p.name.replace("_0000.nii.gz",".nii.gz") for p in images.glob("*_0000.nii.gz")};cases=set()
if len(names)!=100 or {p.name for p in predictions.glob("*.nii.gz")}!=names: raise SystemExit("FreeTumor hidden inventory mismatch")
if len(terminals)!=100: raise SystemExit("FreeTumor hidden case-terminal coverage mismatch")
for index,path in enumerate(terminals):
    x=json.load(open(path,encoding="utf-8"))
    if (x.get("status")!="COMPLETE" or x.get("dataset_role")!="hidden" or x.get("shard_index")!=index
        or x.get("num_shards")!=100 or x.get("case_count")!=1
        or x.get("checkpoint",{}).get("sha256")!=expected
        or x.get("public_images_read") or x.get("public_labels_read") or not x.get("hidden_read")
        or x.get("submission_created") or x.get("uploaded")):
        raise SystemExit(f"invalid FreeTumor hidden shard terminal {index}")
    cases.update(x.get("cases",[]))
if cases!=names: raise SystemExit("FreeTumor hidden shard coverage mismatch")
PY
fi

unzip -oq "${anchor_archive}" -d "${hidden}/anchor"
materialize_args=(--anchor "${hidden}/anchor" --stunet "${hidden}/stunet" \
  --configuration "${configuration}" --output "${hidden}/ensemble" --workers 4)
if [[ "${needs_support}" -eq 1 ]]; then materialize_args+=(--freetumor "${free_hidden}");fi
"${python_bin}" -u "${stunet_root}/pancancer/materialize_stunet_gf_ensemble.py" \
  "${materialize_args[@]}" > "${hidden}/materialize.log" 2>&1
[[ $(find "${hidden}/ensemble" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

tag="${STUNET_ARCHIVE_TAG:-gf_stu_e${epoch}_$(echo "${configuration}" | sed 's/gf_union_stu_//;s/gf_union_stu//;s/_//g')_0825}"
archive="${project_root}/submissions/${tag}.zip"
[[ ${#tag} -lt 59 ]]
"${python_bin}" -u "${project_root}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden}/ensemble" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" > "${archive}.sha256"
printf 'epoch\tconfiguration\tpublic_DSC\tpublic_NSD\tsubmission\tsha256\n' > "${output}/saved_submission.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${epoch}" "${configuration}" "${dsc}" "${nsd}" "${archive}" "${digest}" \
  >> "${output}/saved_submission.tsv"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "public-qualified STU-Net ensemble reproduced label-blind on 100 hidden images" \
  "${configuration}" "${epoch}" "${archive}"
trap - EXIT
