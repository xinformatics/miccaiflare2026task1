#!/usr/bin/env bash
set -euo pipefail

# Save-only consumer for the single predeclared long-trajectory checkpoint and
# deployment selected on fixed public data. Hidden labels are never referenced;
# this script performs inference, identical label-blind postprocessing, and
# exact-100 packaging, but has no upload command.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
controller_root="${root}/public_sweeps/gf_focus_pu_long30_e283_to313_20260825"
controller_terminal="${controller_root}/TERMINAL.json"
public="${root}/public_sweeps/gf_focus_pu_long30_trajectory_20260825"
public_terminal="${public}/TERMINAL.json"
summary="${public}/summary.tsv"
run_dir="${root}/runs/gf_focus_pu_long30_e283_to313_20260825"
checkpoint=""
epoch=""
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
gfconf_archive="${project}/submissions/gf_conf_cbg41_e283.zip"
gfconf_sha256="d8f5f18e860ffd3d0a1dfed44112f0983258320c14f11bf64ab5c1ffd6227e4a"
saved="${root}/public_sweeps/gf_focus_pu_long30_trajectory_saved_20260825"
hidden=""
terminal="${saved}/TERMINAL.json"
hidden_started="${saved}/HIDDEN_INFERENCE_STARTED.json"
lock="${root}/runs/save_long_focus_pu_after_public_20260825.lock"

mkdir -p "${saved}" "${project}/submissions" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "long Focus PU saver already active" >&2; exit 1; }

write_terminal() {
  local status="$1" decision="$2" reason="$3" variant="${4:-}" archive="${5:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${variant}" "${epoch:-}" \
    "${controller_terminal}" "${public_terminal}" "${summary}" "${checkpoint}" \
    "${archive}" "${hidden_started}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,variant,epoch=sys.argv[1:6]
controller,public,summary=map(Path,sys.argv[6:9]);checkpoint=Path(sys.argv[9]) if sys.argv[9] else None;archive=Path(sys.argv[10]) if sys.argv[10] else None;hidden_started=Path(sys.argv[11]);output=Path(sys.argv[12])
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
 "controller_terminal":asset(controller),"public_terminal":asset(public),
 "public_summary":asset(summary),"checkpoint":asset(checkpoint),"submission":archive_asset,
 "hidden_inference_started":asset(hidden_started),
 "public_read":public.is_file(),"hidden_read":hidden_started.is_file() or archive is not None,
 "hidden_labels_read":False,"submission_created":archive is not None,"uploaded":False,
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
    "long Focus saver failed at line ${line}, status ${status}"
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
      printf '%s preserved failed long Focus saver terminal for resume: %s\n' \
        "$(date -Ins)" "${preserved_terminal}"
      ;;
    *) echo "unexpected existing long Focus saver status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while :; do
  if [[ ! -s "${controller_terminal}" ]]; then
    printf '%s waiting for conditional long Focus controller terminal\n' "$(date -Ins)"
    sleep 30
    continue
  fi
  controller_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${controller_terminal}")
  case "${controller_status}" in
    COMPLETE|SKIPPED) break ;;
    FAILED)
      printf '%s waiting for replacement of failed long Focus controller terminal\n' \
        "$(date -Ins)"
      sleep 30
      ;;
    *) echo "unexpected long Focus controller status: ${controller_status}" >&2; exit 2 ;;
  esac
done
controller_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["decision"])' \
  "${controller_terminal}")
if [[ "${controller_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]; then
  write_terminal SKIPPED "${controller_decision}" \
    "conditional long Focus branch did not produce a public-qualified checkpoint"
  trap - EXIT
  exit 0
fi

for file in "${public_terminal}" "${summary}" "${prompt_config}" "${gfconf_archive}"; do
  [[ -s "${file}" ]] || { echo "missing long-save asset: ${file}" >&2; exit 2; }
done
read -r epoch variant minimum dsc nsd checkpoint < <("${python_bin}" - \
  "${controller_terminal}" "${public_terminal}" "${summary}" "${run_dir}" <<'PY'
import csv,hashlib,json,sys
from pathlib import Path
controller_path,public_path,summary_path,run_dir=map(Path,sys.argv[1:])
controller=json.load(open(controller_path,encoding="utf-8"));public=json.load(open(public_path,encoding="utf-8"));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if controller.get("status")!="COMPLETE" or controller.get("decision")!="QUALIFY_SUBMISSION_PREPARATION": raise SystemExit("controller did not qualify")
identity=controller.get("public_terminal",{})
if Path(identity.get("path","")).resolve()!=public_path.resolve() or identity.get("sha256")!=sha(public_path): raise SystemExit("controller/public terminal identity mismatch")
if public.get("status")!="COMPLETE" or public.get("decision")!="QUALIFY_SUBMISSION_PREPARATION": raise SystemExit("public evaluator did not qualify")
if public.get("predeclared_epochs")!=[293,303,313] or public.get("selected_epoch") not in {293,303,313}: raise SystemExit("invalid selected trajectory epoch")
if public.get("selected_variant") not in {"deployed500","deployed1000","touching"}: raise SystemExit("invalid selected deployment")
epoch=int(public["selected_epoch"]);checkpoint=run_dir/f"checkpoint_epoch_{epoch:04d}.pt"
for key,path in (("checkpoint",checkpoint),("summary",summary_path)):
    item=public.get(key,{})
    if Path(item.get("path","")).resolve()!=path.resolve() or item.get("sha256")!=sha(path): raise SystemExit(f"public {key} identity mismatch")
final=run_dir/"checkpoint_epoch_0313.pt";final_identity=controller.get("final_checkpoint",{})
if Path(final_identity.get("path","")).resolve()!=final.resolve() or final_identity.get("sha256")!=sha(final): raise SystemExit("controller final checkpoint identity mismatch")
child_terminal=public_path.parent/f"epoch_{epoch:04d}"/"TERMINAL.json";child=json.load(open(child_terminal,encoding="utf-8"))
child_identity=public.get("selected_public_terminal",{})
if Path(child_identity.get("path","")).resolve()!=child_terminal.resolve() or child_identity.get("sha256")!=sha(child_terminal): raise SystemExit("selected child terminal identity mismatch")
if child.get("status")!="COMPLETE" or child.get("decision")!="QUALIFY_SUBMISSION_PREPARATION" or child.get("selected_variant")!=public["selected_variant"]: raise SystemExit("selected child decision mismatch")
child_checkpoint=child.get("checkpoint",{})
if Path(child_checkpoint.get("path","")).resolve()!=checkpoint.resolve() or child_checkpoint.get("sha256")!=sha(checkpoint): raise SystemExit("selected child checkpoint mismatch")
child_summary=public_path.parent/f"epoch_{epoch:04d}"/"summary.tsv";child_summary_identity=child.get("summary",{})
if Path(child_summary_identity.get("path","")).resolve()!=child_summary.resolve() or child_summary_identity.get("sha256")!=sha(child_summary): raise SystemExit("selected child summary mismatch")
if child.get("hidden_read") is not False or child.get("submission_created") is not False or child.get("uploaded") is not False: raise SystemExit("selected child provenance mismatch")
rows=list(csv.DictReader(open(summary_path,encoding="utf-8"),delimiter="\t"));selected=[row for row in rows if int(row["checkpoint_epoch"])==epoch and row["variant"]==public["selected_variant"]]
if len(selected)!=1: raise SystemExit("selected summary row mismatch")
row=selected[0];dsc=float(row["DSC"]);nsd=float(row["NSD"])
if row["gate"]!="PROMOTE" or dsc<0.6154316379500749 or nsd<0.5104225107769019: raise SystemExit("selected row missed public gate")
if float(public.get("selected_fixed_public_DSC",-1))!=dsc or float(public.get("selected_fixed_public_NSD",-1))!=nsd: raise SystemExit("selected public metric mismatch")
if public.get("hidden_read") is not False or public.get("submission_created") is not False or public.get("uploaded") is not False: raise SystemExit("public-only provenance mismatch")
minimum=500 if row["variant"]=="deployed500" else 1000
print(epoch,row["variant"],minimum,dsc,nsd,checkpoint.resolve())
PY
)
[[ -s "${checkpoint}" ]] || { echo "selected trajectory checkpoint is missing" >&2; exit 2; }
hidden="${root}/hidden_predictions/gf_focus_pu_long30_e${epoch}_20260825"

# From this point onward the script may inspect hidden images or hidden-derived
# supports. Persist that transition first so even an interrupted/failed run has
# truthful label-blind provenance in its terminal.
"${python_bin}" - "${variant}" "${checkpoint}" "${hidden_started}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
variant,checkpoint,output=sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3])
payload={
 "schema_version":1,
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "selected_variant":variant,
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
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
  printf '%s waiting for long Focus hidden GPU: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
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
  infer_shard 0 0 & pid0=$!; infer_shard 1 1 & pid1=$!
  status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?; [[ "${status}" -eq 0 ]]
elif [[ "${free0}" -eq 1 ]]; then
  infer_shard 0 0; infer_shard 0 1
else
  infer_shard 1 0; infer_shard 1 1
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
  deployed500) tag="gf_pu${epoch}_d500_0825" ;;
  deployed1000) tag="gf_pu${epoch}_d1k_0825" ;;
  touching) tag="gf_pu${epoch}_touch_0825" ;;
  *) exit 2 ;;
esac
archive="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${selected}" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
digest=$(sha256sum "${archive}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${archive}" > "${archive}.sha256"
printf 'tag\tvariant\tcheckpoint\tDSC\tNSD\tsubmission\tsha256\n' > "${saved}/saved_submission.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${variant}" "${checkpoint}" \
  "${dsc}" "${nsd}" "${archive}" "${digest}" >> "${saved}/saved_submission.tsv"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "public-selected long-trajectory deployment reproduced label-blind on 100 hidden images" \
  "${variant}" "${archive}"
trap - EXIT
cat "${saved}/saved_submission.tsv"
