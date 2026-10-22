#!/usr/bin/env bash
set -euo pipefail

# Frozen public replay and conditional hidden packaging for the exact-707
# source-STU confirmer. The classifier threshold and seed deployment mode come
# only from grouped training OOF. Public labels gate one candidate; hidden
# labels are unreachable and no upload operation exists.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
stunet_root="${project}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
oof_terminal="${root}/analysis/source_stunet_verified707_fullvolume_oof_20260827/FULL_VOLUME_OOF_TERMINAL.json"
patch_gate="${root}/analysis/source_stunet_verified707_refiner_20260827/PATCH_OOF_GATE.json"
source_public="${stunet_root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826/TERMINAL.json"
source_model="${project}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceStratified__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
public_images="${project}/validation/Validation-Public-Images"
public_labels="${project}/validation/Validation-Public-Labels"
public_anchor="${root}/analysis/lesion_error_touchcomp_20260825/predictions"
e160_stunet="${stunet_root}/public_sweeps/stunetb_matched_e160_20260826/component_predictions"
freetumor_public="${project}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
public_output="${root}/public_sweeps/source_stunet_refiner_20260827"
public_base="${public_output}/epoch160_p70_base"
public_work="${public_output}/work_bbox24"
public_candidate="${public_output}/candidate"
public_base_metrics="${public_output}/epoch160_p70_base.metrics.csv"
public_candidate_metrics="${public_output}/candidate.metrics.csv"
public_robust="${public_output}/paired_vs_epoch160_p70.robust.json"
public_terminal="${public_output}/PUBLIC_TERMINAL.json"

hidden_images="${project}/validation/Validation-Hidden-Images"
champion_saved="${stunet_root}/public_sweeps/stunetb_component_e160_p70_saved_20260826/TERMINAL.json"
champion_archive="${project}/submissions/gf_stu_e160_gfanchorconfirmeitherp70_0825.zip"
champion_sha="6f84ec34f53c8a97ddf62fbae7148d34507b61be3b838f3d263ae408de25456a"
hidden="${root}/hidden_predictions/source_stunet_refiner_20260827"
hidden_source="${hidden}/source_stunet"
hidden_base="${hidden}/epoch160_p70_base"
hidden_work="${hidden}/work_bbox24"
hidden_candidate="${hidden}/candidate"
archive="${project}/submissions/gf_ssr_0827.zip"
terminal="${public_output}/TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_source_stunet_b32_fold%s_20260827' "${root}" "$1"; }

validate_mask_inventory() {
  local images="$1" masks="$2" expected_count="$3"
  "${python_bin}" - "${images}" "${masks}" "${expected_count}" <<'PY'
import sys
from pathlib import Path
images, masks = map(Path, sys.argv[1:3]); expected_count = int(sys.argv[3])
expected = {
    path.name.removesuffix("_0000.nii.gz") + ".nii.gz"
    for path in images.glob("*_0000.nii.gz")
}
observed = {path.name for path in masks.glob("*.nii.gz") if path.is_file()}
if len(expected) != expected_count or observed != expected:
    raise SystemExit(
        f"mask inventory mismatch: expected={len(expected)} observed={len(observed)} "
        f"missing={sorted(expected-observed)[:5]} extra={sorted(observed-expected)[:5]}"
    )
if any((masks / name).stat().st_size < 1 for name in observed):
    raise SystemExit("empty prediction file")
PY
}

mkdir -p "${public_output}" "${hidden}" "${project}/submissions"
exec 9>"${root}/runs/run_source_stunet_refiner_public_save_20260827.lock"
flock -n 9 || { echo "source-STU refiner public/save already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  [[ "${status}" == COMPLETE || "${status}" == SKIPPED ]] && exit 0
  [[ "${status}" == FAILED ]] && mv --no-clobber "${terminal}" "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3" archive_path="${4:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${archive_path}" \
    "${oof_terminal}" "${patch_gate}" "${source_public}" "${public_terminal}" "${archive}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
status,decision,reason,archive_text=sys.argv[1:5]; oof,gate,source,public,expected_archive,output=map(Path,sys.argv[5:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
submission=None
if archive_text:
 archive=Path(archive_text)
 if archive.resolve()!=expected_archive.resolve() or not archive.is_file() or len(archive.name)>=64: raise SystemExit("invalid source-STU archive")
 with zipfile.ZipFile(archive) as handle:
  members=[item for item in handle.infolist() if not item.is_dir()]
  if (len(members)!=100 or len({item.filename for item in members})!=100
      or any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size<1 for item in members)
      or handle.testzip() is not None): raise SystemExit("invalid source-STU ZIP inventory")
 submission={"path":str(archive.resolve()),"sha256":sha(archive),"members":100}
public_payload=json.load(open(public,encoding="utf-8")) if public.is_file() else {}
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,"oof":asset(oof),"patch_gate":asset(gate),
 "source_public_selection":asset(source),"fixed_public":asset(public),"submission":submission,
 "public_metrics":public_payload.get("candidate_metrics"),"public_read":public.is_file(),
 "hidden_inputs_read":submission is not None,"hidden_labels_read":False,
 "submission_created":submission is not None,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
}
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    write_terminal FAILED RETRY_FROM_VALIDATED_PUBLIC_OR_HIDDEN_CACHE "failed at line ${line}, status ${status}"
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

while [[ ! -s "${oof_terminal}" || ! -s "${source_public}" ]]; do
  printf '%s waiting for source-STU OOF qualification and source public endpoint\n' "$(date -Ins)"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != QUALIFY_STUNET_SEED_FIXED_PUBLIC ]]; then
  [[ "${oof_decision}" == REJECT_STUNET_SEED_FULL_VOLUME_OOF || "${oof_decision}" == REJECT_STUNET_SEED_PATCH_GATE ]]
  write_terminal SKIPPED KEEP_EPOCH160_P70 "source-STU confirmer failed training-only full-volume qualification"
  trap - EXIT; exit 0
fi
read -r epoch checkpoint checkpoint_sha classification_threshold < <("${python_bin}" - \
  "${oof_terminal}" "${patch_gate}" "${source_public}" "${source_model}" <<'PY'
import hashlib,json,sys
from pathlib import Path
oof_path,gate_path,source_path,model=map(Path,sys.argv[1:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); o=json.load(open(oof_path)); g=json.load(open(gate_path)); s=json.load(open(source_path)); selected=s.get("selected") or {}
if (o.get("status")!="COMPLETE" or o.get("decision")!="QUALIFY_STUNET_SEED_FIXED_PUBLIC"
    or o.get("public_read") or o.get("hidden_read") or g.get("decision")!="QUALIFY_STUNET_SEED_FULL_VOLUME_OOF"
    or g.get("deployment_mode")!="seed" or not all(g.get("checks",{}).values())
    or s.get("status")!="COMPLETE" or s.get("predeclared_epochs")!=[40,80]
    or s.get("hidden_read") or s.get("submission_created") or s.get("uploaded")):
 raise SystemExit("invalid source-STU public prerequisite")
epoch=int(selected.get("epoch",-1)); checkpoint=Path(selected.get("checkpoint",{}).get("path","")); expected=(model/"fold_all"/f"checkpoint_inference_epoch_{epoch:04d}.pth").resolve()
if epoch not in (40,80) or checkpoint.resolve()!=expected or selected.get("checkpoint",{}).get("sha256")!=sha(checkpoint): raise SystemExit("invalid source checkpoint")
print(epoch,checkpoint,sha(checkpoint),f'{float(g["classification_threshold"]):.2f}')
PY
)
public_source="${stunet_root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826/epoch_$(printf '%04d' "${epoch}")/predictions"
for directory in "${public_images}" "${public_labels}" "${public_anchor}" "${e160_stunet}" "${freetumor_public}" "${public_source}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || { echo "invalid public inventory: ${directory}" >&2; exit 2; }
done

if [[ $(find "${public_base}" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -ne 50 ]]; then
  "${python_bin}" -u "${stunet_root}/materialize_stunet_gf_ensemble.py" \
    --anchor "${public_anchor}" --stunet "${e160_stunet}" --freetumor "${freetumor_public}" \
    --configuration gf_anchor_confirm_either_p70 --output "${public_base}" --workers 4 \
    >"${public_output}/materialize_epoch160_p70.log" 2>&1
fi
validate_mask_inventory "${public_images}" "${public_base}" 50
if [[ ! -s "${public_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${public_images}" --proposal "source_stunet=${public_source}" \
    --component-policy source_separate --base "${public_base}" --work "${public_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm 24 --tile-stride-mm 72 --resume >"${public_output}/prepare.log" 2>&1
fi
checkpoint_args=()
for fold in 0 1 2 3 4; do
  refiner="$(fold_dir "${fold}")/checkpoint_best_classification.pt"; [[ -s "${refiner}" ]]
  checkpoint_args+=(--checkpoint "${refiner}")
done
if ! "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${public_work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" >/dev/null 2>&1; then
  while :; do
    used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
    if [[ "${used0}" -le 1024 ]]; then inference_gpu=0; break; fi
    if [[ "${used1}" -le 1024 ]]; then inference_gpu=1; break; fi
    sleep 30
  done
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${public_work}" "${checkpoint_args[@]}" --batch-size 32 --workers 2 --resume \
    >"${public_output}/infer.log" 2>&1
fi
"${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
  --marker "${public_work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" >/dev/null
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
  --work "${public_work}" --base "${public_base}" --output "${public_candidate}" --resume \
  --classification-threshold "${classification_threshold}" --segmentation-threshold 0.50 \
  --mode refined --source-mode source_stunet=seed --min-seed-coverage 0 \
  --min-refined-overlap 0 --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
  >"${public_output}/materialize_candidate.log" 2>&1
validate_mask_inventory "${public_images}" "${public_candidate}" 50
"${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${public_base}" --gt "${public_labels}" \
  --csv "${public_base_metrics}" --workers 6 >"${public_output}/base.score.log" 2>&1
"${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${public_candidate}" --gt "${public_labels}" \
  --csv "${public_candidate_metrics}" --workers 6 >"${public_output}/candidate.score.log" 2>&1
"${python_bin}" -u "${root}/select_checkpoint_robust.py" --anchor "${public_base_metrics}" \
  --candidate "${public_candidate_metrics}" --anchor-name epoch160_p70_champion \
  --candidate-name epoch160_p70_plus_source_stunet_confirmer --min-dsc-gain 0.015 \
  --min-nsd-gain 0.0 --min-win-rate 0.52 --max-catastrophic-rate 0.10 \
  --bootstrap 20000 --seed 20260827 --output "${public_robust}" >"${public_output}/robust.log" 2>&1
"${python_bin}" - "${epoch}" "${checkpoint}" "${checkpoint_sha}" "${classification_threshold}" \
  "${public_base_metrics}" "${public_candidate_metrics}" "${public_robust}" "${public_terminal}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
epoch=int(sys.argv[1]); checkpoint=Path(sys.argv[2]); expected=sys.argv[3]; threshold=float(sys.argv[4]); base,candidate,robust_path,output=map(Path,sys.argv[5:]); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def means(path):
 rows=list(csv.DictReader(open(path,encoding="utf-8"))); return {k:sum(float(r[k]) for r in rows)/len(rows) for k in ("Lesion_DSC","Lesion_NSD")}
b=means(base); c=means(candidate); r=json.load(open(robust_path,encoding="utf-8"))
if abs(b["Lesion_DSC"]-0.6216639257651584)>5e-5 or abs(b["Lesion_NSD"]-0.5347520263176905)>5e-5: raise SystemExit(f"epoch160 p70 public base drifted: {b}")
qualified=(r.get("decision")=="PROMOTE" and c["Lesion_DSC"]>=0.6366639257651584 and c["Lesion_NSD"]>=0.5347520263176905)
p={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"status":"COMPLETE",
 "decision":"QUALIFY_HIDDEN_SAVE" if qualified else "KEEP_EPOCH160_P70",
 "selected_source_epoch":epoch,"source_checkpoint":{"path":str(checkpoint.resolve()),"sha256":expected},
 "classification_threshold":threshold,"deployment_mode":"seed","base_metrics":b,"candidate_metrics":c,
 "required_absolute":{"Lesion_DSC":0.6366639257651584,"Lesion_NSD":0.5347520263176905},
 "paired_gate":{"path":str(robust_path.resolve()),"sha256":sha(robust_path),"decision":r.get("decision")},
 "public_read":True,"hidden_read":False,"submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp");t.write_text(json.dumps(p,indent=2)+"\n");os.replace(t,output)
PY
public_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" != QUALIFY_HIDDEN_SAVE ]]; then
  write_terminal COMPLETE KEEP_EPOCH160_P70 "trained source-STU addition missed the +0.015 DSC/NSD public gate"
  trap - EXIT; exit 0
fi

"${python_bin}" - "${champion_saved}" "${champion_archive}" "${champion_sha}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,archive=map(Path,sys.argv[1:3]); expected=sys.argv[3]; p=json.load(open(terminal,encoding="utf-8")); sha=hashlib.sha256(archive.read_bytes()).hexdigest()
if (p.get("status")!="COMPLETE" or p.get("decision")!="ARCHIVE_READY_FOR_MANUAL_UPLOAD"
    or p.get("selected_epoch")!=160 or p.get("selected_configuration")!="gf_anchor_confirm_either_p70"
    or p.get("submission",{}).get("path")!=str(archive.resolve()) or p.get("submission",{}).get("sha256")!=expected
    or sha!=expected or p.get("hidden_labels_read") or p.get("uploaded")):
 raise SystemExit("invalid epoch160 p70 hidden base")
PY
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
mkdir -p "${hidden_source}" "${hidden_base}"
while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used0}" -le 1024 && "${used1}" -le 1024 ]] && break
  sleep 30
done
predict_shard() {
  local gpu="$1" shard="$2"
  "${python_bin}" -u "${stunet_root}/predict_stunet_safe.py" --images "${hidden_images}" \
    --output "${hidden_source}" --model-dir "${source_model}" --checkpoint "$(basename "${checkpoint}")" \
    --gpu "${gpu}" --part-id "${shard}" --num-parts 2 --workers 1 --sequential \
    --dataset-role hidden --terminal "${hidden}/SOURCE_STUNET_SHARD_${shard}.json" \
    >"${hidden}/source_stunet_shard_${shard}.log" 2>&1
}
predict_shard 0 0 & p0=$!; predict_shard 1 1 & p1=$!; status=0
wait "${p0}" || status=$?; wait "${p1}" || status=$?; [[ "${status}" -eq 0 ]]
"${python_bin}" - "${hidden_images}" "${hidden_source}" \
  "${hidden}/SOURCE_STUNET_SHARD_0.json" "${hidden}/SOURCE_STUNET_SHARD_1.json" \
  "${checkpoint}" "${checkpoint_sha}" <<'PY'
import hashlib,json,sys
from pathlib import Path
images,predictions,shard0,shard1,checkpoint=map(Path,sys.argv[1:6]); expected_sha=sys.argv[6]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={p.name.removesuffix("_0000.nii.gz")+".nii.gz" for p in images.glob("*_0000.nii.gz")}
observed={p.name for p in predictions.glob("*.nii.gz") if p.is_file()}
if len(expected)!=100 or observed!=expected or sha(checkpoint)!=expected_sha:
 raise SystemExit("hidden source-STU coverage/checkpoint mismatch")
terminal_cases=set()
for shard_id,terminal in enumerate((shard0,shard1)):
 p=json.load(open(terminal,encoding="utf-8"))
 cases=set(p.get("cases") or []); terminal_cases.update(cases)
 if (p.get("status")!="COMPLETE" or p.get("dataset_role")!="hidden"
     or p.get("part_id")!=shard_id or p.get("num_parts")!=2
     or p.get("case_count")!=50 or len(cases)!=50
     or not p.get("hidden_read") or p.get("training_images_read")
     or p.get("public_images_read") or p.get("public_labels_read")
     or p.get("hidden_labels_read") or p.get("save_probabilities")
     or not p.get("sequential") or p.get("submission_created") or p.get("uploaded")
     or p.get("checkpoint",{}).get("path")!=str(checkpoint.resolve())
     or p.get("checkpoint",{}).get("sha256")!=expected_sha):
  raise SystemExit(f"invalid hidden source-STU shard {shard_id}")
if terminal_cases!=expected:
 raise SystemExit("hidden source-STU shard case union mismatch")
PY
unzip -oq "${champion_archive}" -d "${hidden_base}"
validate_mask_inventory "${hidden_images}" "${hidden_base}" 100
if [[ ! -s "${hidden_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare --images "${hidden_images}" \
    --proposal "source_stunet=${hidden_source}" --component-policy source_separate \
    --base "${hidden_base}" --work "${hidden_work}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 --resume \
    >"${hidden}/prepare.log" 2>&1
fi
while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used0}" -le 1024 ]]; then inference_gpu=0; break; fi
  if [[ "${used1}" -le 1024 ]]; then inference_gpu=1; break; fi
  sleep 30
done
if ! "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${hidden_work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${hidden_work}" "${checkpoint_args[@]}" --batch-size 32 --workers 2 --resume \
    >"${hidden}/infer.log" 2>&1
fi
"${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
  --marker "${hidden_work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" >/dev/null
"${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize --work "${hidden_work}" \
  --base "${hidden_base}" --output "${hidden_candidate}" --resume \
  --classification-threshold "${classification_threshold}" --segmentation-threshold 0.50 \
  --mode refined --source-mode source_stunet=seed --min-seed-coverage 0 \
  --min-refined-overlap 0 --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
  >"${hidden}/materialize.log" 2>&1
validate_mask_inventory "${hidden_images}" "${hidden_candidate}" 100
"${python_bin}" -u "${project}/nnunet/package_codabench.py" --images "${hidden_images}" \
  --pred-dir "${hidden_candidate}" --output "${archive}" --expected-count 100 >"${hidden}/package.log" 2>&1
sha256sum "${archive}" >"${archive}.sha256"
write_terminal COMPLETE CANDIDATE_SAVED "passed exact-707 OOF and +0.015 fixed-public gates" "${archive}"
trap - EXIT
