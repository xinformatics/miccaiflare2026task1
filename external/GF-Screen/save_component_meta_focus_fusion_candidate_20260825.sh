#!/usr/bin/env bash
set -euo pipefail

# Label-blind hidden consumer for a component-meta candidate that passed both
# exact-707 OOF and the fixed public gate. Produces and audits one flat 100-mask
# ZIP, but never uploads it.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
public="${root}/public_sweeps/component_meta_focus_fusion_20260825"
public_terminal="${public}/PUBLIC_COMPONENT_TERMINAL.json"
oof="${root}/analysis/component_meta_focus_fusion_fullvolume_oof_20260824"
calibration="${oof}/deployment_calibration.json"
component_decision="${root}/analysis/component_meta_classifier_frozen_focus_fusion_20260824/DECISION.json"
patch_fusion_decision="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
contour_summary="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_patch_oof.json"
selection="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/checkpoint_selection.json"
plan="${biomed}/analysis/weak_domain_biomedparse_707_selected_20260823/SELECTION_PLAN.json"
margin_decision="${root}/analysis/weak707_bbox_margin_headroom_20260823/DECISION.json"
cell_decision="${root}/analysis/lung_cell707_scale_20260823/DECISION.json"
hidden_images="${project}/validation/Validation-Hidden-Images"
hidden_inputs="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_direct_whole_body_lesion_e030_confirm_distance_mm_0_rescue_min_mm3_inf/inputs"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/hidden_raw"
champion="${root}/hidden_predictions/cbg_intersection_20260822/confirm_exact"
cells="${root}/hidden_predictions/candidate_refiner_v4_cell707_20260823/cells48"
gf_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
output="${root}/hidden_predictions/component_meta_focus_fusion_20260825"
epoch_root="${output}/epoch105_abd_g090_l050"
epoch105="${epoch_root}/predictions"
work="${output}/work_bbox24_ss_v4_cell48"
override="${output}/component_probabilities.npz"
grid="${output}/selected_grid.json"
prediction_root="${output}/predictions"
tag="gf_cmff_0825"
prediction="${prediction_root}/${tag}/fold0"
submission="${project}/submissions/${tag}.zip"
packager="${project}/nnunet/package_codabench.py"
manifest="${output}/saved_component_meta_submission.tsv"
terminal="${output}/HIDDEN_COMPONENT_TERMINAL.json"
lock="${root}/runs/save_component_meta_focus_fusion_candidate_20260825.lock"

mkdir -p "${output}" "${epoch105}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "component-meta hidden consumer already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED|SKIPPED_PUBLIC)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.${existing_status,,}_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected component-meta hidden terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3" archive="${4:-}" digest="${5:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${public_terminal}" \
    "${calibration}" "${work}/AUDIT.json" "${override%.npz}.json" \
    "${archive}" "${digest}" "${manifest}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
public,calibration,audit,override=map(Path,sys.argv[4:8])
archive,digest=sys.argv[8:10]; manifest,output=map(Path,sys.argv[10:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "public_terminal":asset(public),"calibration":asset(calibration),
 "inference_audit":asset(audit),"component_override":asset(override),
 "hidden_inputs_read":audit.is_file(),"hidden_labels_read":False,
 "submission_created":bool(archive),"submission":archive or None,
 "submission_sha256":digest or None,"saved_manifest":asset(manifest),"uploaded":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED NO_SUBMISSION \
    "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for fixed-public component terminal\n' "$(date -Ins)"
  sleep 30
done
public_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${public_terminal}")
if [[ "${public_decision}" != QUALIFY_FOR_HIDDEN ]]; then
  write_terminal SKIPPED_PUBLIC NO_SUBMISSION \
    "fixed-public component decision was ${public_decision}"
  trap - EXIT
  exit 0
fi
for required in "${calibration}" "${component_decision}" "${patch_fusion_decision}" \
    "${contour_summary}" "${selection}" "${plan}" "${margin_decision}" \
    "${cell_decision}" "${gf_checkpoint}" "${packager}" \
    "${cells}/PROPOSALS_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${hidden_images}" "${hidden_inputs}" "${currentgf}" "${champion}"; do
  [[ -d "${directory}" ]] || { echo "missing required directory: ${directory}" >&2; exit 2; }
done
[[ $(find "${hidden_images}" -maxdepth 1 -name '*_0000.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${hidden_inputs}" -maxdepth 1 -name '*.npz' | wc -l) -eq 100 ]]
[[ $(find "${currentgf}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${champion}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
for colour in 0 1 2 3 4 5 6 7; do
  [[ $(find "${cells}/colour${colour}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
done
"${python_bin}" - "${public_terminal}" "${component_decision}" \
  "${patch_fusion_decision}" "${cell_decision}" <<'PY'
import json,sys
public,component,patch,cell=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if public.get("status")!="COMPLETE" or public.get("decision")!="QUALIFY_FOR_HIDDEN":
 raise SystemExit("fixed-public gate is not complete and qualified")
if public.get("hidden_read") is not False or public.get("submission_created") is not False:
 raise SystemExit("fixed-public provenance drift")
if component.get("decision")!="QUALIFY_COMPONENT_META_CLASSIFIER":
 raise SystemExit("component classifier decision drift")
if patch.get("decision")!="QUALIFY_FOCUS_CLASSIFIER_FUSION":
 raise SystemExit("patch fusion decision drift")
if cell.get("decision")!="PROMOTE_LUNG_CELLS_707":
 raise SystemExit("lung-cell proposal decision drift")
PY
bbox_margin=$("${python_bin}" - "${margin_decision}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
value=int(p.get("effective_margin_mm",p["baseline_margin_mm"] if p["decision"]=="KEEP_MARGIN_24" else p["selected"]["margin_mm"]))
if value not in {16,24,32,48,72,96}: raise SystemExit("invalid frozen bbox margin")
print(value)
PY
)

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_any_gpu() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    free0=0; free1=0
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]] && free0=1
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]] && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}

epoch_marker="${epoch_root}/PROPOSALS_COMPLETE.json"
if [[ ! -s "${epoch_marker}" ]]; then
  wait_any_gpu
  mkdir -p "${epoch_root}/gpu0" "${epoch_root}/gpu1"
  run_epoch_shard() {
    local gpu="$1" shard="$2"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${hidden_images}" --save_prediction_path "${epoch_root}/gpu${shard}" \
      --trained_pth "${gf_checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 --focus_batch_size 16 \
      --workers 2 --num_shards 2 --shard_id "${shard}" --skip_existing \
      > "${output}/epoch105_gpu${shard}.log" 2>&1
  }
  if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
    run_epoch_shard 0 0 & p0=$!; run_epoch_shard 1 1 & p1=$!
    status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
    [[ "${status}" -eq 0 ]] || exit "${status}"
  elif [[ "${free0}" -eq 1 ]]; then
    run_epoch_shard 0 0; run_epoch_shard 0 1
  else
    run_epoch_shard 1 0; run_epoch_shard 1 1
  fi
  cp "${epoch_root}/gpu0/"*.nii.gz "${epoch105}/"
  cp "${epoch_root}/gpu1/"*.nii.gz "${epoch105}/"
  checkpoint_sha=$(sha256sum "${gf_checkpoint}" | awk '{print $1}')
  "${python_bin}" - "${hidden_images}" "${epoch105}" "${epoch_marker}" \
    "${gf_checkpoint}" "${checkpoint_sha}" <<'PY'
import datetime,json,sys
from pathlib import Path
images,predictions,marker,checkpoint=map(Path,sys.argv[1:5]); digest=sys.argv[5]
expected={path.name[:-12]+".nii.gz" for path in images.glob("*_0000.nii.gz")}
observed={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=100 or observed!=expected: raise SystemExit("hidden epoch105 coverage mismatch")
marker.write_text(json.dumps({"created_at":datetime.datetime.now().astimezone().isoformat(),
 "cases":100,"labels_read":False,"checkpoint":str(checkpoint.resolve()),
 "checkpoint_sha256":digest,"window_hu":[-175,250],"glance_threshold":0.90,
 "lesion_threshold":0.50},indent=2)+"\n",encoding="utf-8")
PY
fi
[[ $(find "${epoch105}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

run_bank() {
  local bank="$1" gpu="$2"
  bash "${biomed}/run_selected_weak_prompt_bank.sh" \
    "${hidden_inputs}" "${hidden_images}" "${output}/${bank}_weak" \
    "${plan}" "${bank}" 100 "${gpu}" > "${output}/${bank}_weak.controller.log" 2>&1
}
wait_any_gpu
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  run_bank autopet 0 & p0=$!; run_bank lndb 1 & p1=$!
  status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  run_bank autopet 0; run_bank lndb 0
else
  run_bank autopet 1; run_bank lndb 1
fi
autopet="${output}/autopet_weak/predictions_nifti"
lndb="${output}/lndb_weak/predictions_nifti"
[[ $(find "${autopet}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]
[[ $(find "${lndb}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
  cell_args=()
  for colour in 0 1 2 3 4 5 6 7; do
    cell_args+=(--proposal "lung_cell${colour}=${cells}/colour${colour}")
  done
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${hidden_images}" --proposal "epoch105=${epoch105}" \
    --proposal "currentgf=${currentgf}" --proposal "autopet_weak=${autopet}" \
    --proposal "lndb_weak=${lndb}" "${cell_args[@]}" \
    --component-policy source_separate --base "${champion}" --work "${work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 \
    --max-mm3 1000000000000 --bbox-margin-mm "${bbox_margin}" \
    --tile-stride-mm 72 --resume > "${output}/prepare_hidden.log" 2>&1
fi

checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
checkpoint_args=(); checkpoints=()
for fold in 0 1 2 3 4; do
  checkpoint="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold${fold}_20260823/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing B32 checkpoint: ${checkpoint}" >&2; exit 2; }
  checkpoint_args+=(--checkpoint "${checkpoint}"); checkpoints+=("${checkpoint}")
done
mapfile -t contours < <("${python_bin}" - "${contour_summary}" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1],encoding="utf-8"))["fold_records"]
if sorted(int(row["fold"]) for row in rows)!=list(range(5)): raise SystemExit("invalid Focus folds")
for row in sorted(rows,key=lambda value:int(value["fold"])): print(row["checkpoint"])
PY
)
contour_mode=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["input_mode"])' "${contour_summary}")
contour_args=()
for contour in "${contours[@]}"; do
  [[ -s "${contour}" ]] || { echo "missing Focus checkpoint: ${contour}" >&2; exit 2; }
  contour_args+=(--contour-checkpoint "${contour}")
done
inference_matches() {
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" \
    "${contour_args[@]}" --contour-input-mode "${contour_mode}" \
    --classifier-fusion "${patch_fusion_decision}" > /dev/null 2>&1
}
if ! inference_matches; then
  wait_any_gpu
  [[ "${free0}" -eq 1 ]] && inference_gpu=0 || inference_gpu=1
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u \
    "${root}/apply_candidate_refiner.py" infer --work "${work}" \
    "${checkpoint_args[@]}" "${contour_args[@]}" \
    --classifier-fusion "${patch_fusion_decision}" --batch-size 4 --workers 2 --resume \
    > "${output}/infer_hidden.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 4 \
  --require-predictions --require-inference-plan > "${output}/audit_hidden.log" 2>&1

"${python_bin}" -u "${root}/prepare_component_probability_override.py" \
  --work "${work}" --component-decision "${component_decision}" \
  --patch-fusion-decision "${patch_fusion_decision}" --output "${override}" \
  > "${output}/component_probability.log" 2>&1
"${python_bin}" -u "${root}/export_selected_candidate_refiner_grid.py" \
  --calibration "${calibration}" --tag "${tag}" --output "${grid}" \
  > "${output}/selected_grid.log" 2>&1
read -r segmentation deployment min_seed min_overlap min_ratio max_ratio <<< "$(
  "${python_bin}" - "${calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); selected=p["selected"]
seg=float(selected["segmentation_threshold"]); deployment=p.get("calibration_deployment")
if not 0<seg<1 or deployment not in {"standalone","anchor_currentgf"}: raise SystemExit("invalid deployment")
g=p.get("geometry_gates",{}); values=[float(g.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not all(math.isfinite(x) and x>=0 for x in values) or values[0]>1 or values[1]>1 or values[2]>values[3]:
 raise SystemExit("invalid geometry gates")
print(f"{seg:.12g}",deployment,*(f"{x:.12g}" for x in values))
PY
)"
empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/materialize_candidate_refiner_component_grid.py" \
  --work "${work}" --base "${champion}" "${empty_arg[@]}" \
  --output-root "${prediction_root}" --configurations "${grid}" --fold 0 \
  --segmentation-threshold "${segmentation}" --component-probabilities "${override}" \
  --min-seed-coverage "${min_seed}" --min-refined-overlap "${min_overlap}" \
  --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
  --writer-workers 2 > "${output}/materialize_hidden.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

"${python_bin}" -u "${packager}" --images "${hidden_images}" \
  --pred-dir "${prediction}" --output "${submission}" --expected-count 100 \
  > "${output}/package.log" 2>&1
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
[[ ${#tag} -lt 60 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
printf '%s  %s\n' "${digest}" "${submission}" > "${submission}.sha256"
checkpoints_csv=$(IFS=,; echo "${checkpoints[*]}")
printf 'rank\ttag\tfamily\tcheckpoints\tpublic_gate\tsubmission\tsha256\n' > "${manifest}"
printf '1\t%s\tcomponent_meta_focus_fusion\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${checkpoints_csv}" "${public_terminal}" "${submission}" "${digest}" >> "${manifest}"
write_terminal COMPLETE CANDIDATE_SAVED \
  "passed exact-707 OOF and fixed-public gates" "${submission}" "${digest}"
trap - EXIT
printf '%s saved %s (%s)\n' "$(date -Ins)" "${submission}" "${digest}"
