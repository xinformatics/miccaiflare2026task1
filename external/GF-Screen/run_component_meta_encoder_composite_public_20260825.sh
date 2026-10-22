#!/usr/bin/env bash
set -euo pipefail

# Fixed 50-case screen for the exact-707-qualified component-meta/encoder
# composition. Classifier-fusion confidence and encoder-routed contours are
# inferred into separate audited caches and joined only by exact patch ID.
# Hidden inputs and submission creation are unreachable from this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
oof="${root}/analysis/component_meta_encoder_composite_fullvolume_oof_20260825"
oof_terminal="${oof}/TERMINAL.json"
calibration="${oof}/deployment_calibration.json"
component_decision="${root}/analysis/component_meta_classifier_frozen_focus_fusion_20260824/DECISION.json"
patch_fusion_decision="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
contour_summary="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_patch_oof.json"
encoder_routes="${root}/analysis/focus_encoder_upgrade_composite_routes_20260824/DECISION.json"
selection="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/checkpoint_selection.json"
champion="${root}/analysis/DEPLOYMENT_CHAMPION.json"
anchor_metrics="${root}/public_sweeps/gfconf_support_growth_quick_20260824/touching_component.metrics.csv"
champion_predictions="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf"
labels="${project}/validation/Validation-Public-Labels"
classifier_cache="${root}/public_sweeps/component_meta_encoder_classifier_cache_20260825"
classifier_work="${classifier_cache}/work_bbox24_ss_v4_cell48"
output="${root}/public_sweeps/component_meta_encoder_composite_20260825"
encoder_work="${output}/work_bbox24_encoder_routes"
override="${output}/component_probabilities.npz"
grid="${output}/selected_grid.json"
candidate_root="${output}/predictions"
tag="component_meta_encoder_selected"
prediction="${candidate_root}/${tag}/fold0"
metrics="${output}/${tag}.metrics.csv"
score_log="${output}/${tag}.score.log"
gate="${output}/dsc_first_vs_touchcomp.json"
summary="${output}/summary.tsv"
terminal="${output}/TERMINAL.json"
lock="${root}/runs/run_component_meta_encoder_composite_public_20260825.lock"

mkdir -p "${output}" "${classifier_cache}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "component-meta encoder public controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${oof_terminal}" \
    "${calibration}" "${classifier_work}/AUDIT.json" "${encoder_work}/AUDIT.json" \
    "${override%.npz}.json" "${metrics}" "${gate}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]; paths=list(map(Path,sys.argv[4:-1])); output=Path(sys.argv[-1])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path): return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
keys=("oof_terminal","calibration","classifier_audit","encoder_audit","component_override","metrics","gate","summary")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"status":status,"decision":decision,"reason":reason,**{k:asset(p) for k,p in zip(keys,paths)},"public_read":asset(paths[5]) is not None,"hidden_read":False,"submission_created":False,"uploaded":False}
tmp=output.with_name(f".{output.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,output)
PY
}
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_TOUCHCOMP \
    "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for component-meta/encoder exact-707 terminal\n' "$(date -Ins)"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("decision",""))' "${oof_terminal}")
if [[ "${oof_decision}" != QUALIFY_FOR_FIXED_PUBLIC ]]; then
  write_terminal SKIPPED_OOF KEEP_TOUCHCOMP "exact-707 decision was ${oof_decision}"
  trap - EXIT
  exit 0
fi
for required in "${calibration}" "${component_decision}" "${patch_fusion_decision}" \
    "${contour_summary}" "${encoder_routes}" "${selection}" "${champion}" \
    "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${champion_predictions}" "${labels}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done
[[ $(find "${champion_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" - "${oof_terminal}" "${component_decision}" "${patch_fusion_decision}" \
  "${encoder_routes}" "${champion}" <<'PY'
import json,sys
oof,component,patch,routes,champion=(json.load(open(p,encoding="utf-8")) for p in sys.argv[1:])
if oof.get("status")!="COMPLETE" or oof.get("decision")!="QUALIFY_FOR_FIXED_PUBLIC": raise SystemExit("OOF is not qualified")
if oof.get("public_read") is not False or oof.get("hidden_read") is not False: raise SystemExit("OOF provenance contaminated")
if component.get("decision")!="QUALIFY_COMPONENT_META_CLASSIFIER": raise SystemExit("component decision drift")
if patch.get("decision")!="QUALIFY_FOCUS_CLASSIFIER_FUSION": raise SystemExit("classifier fusion drift")
if routes.get("decision")!="QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME": raise SystemExit("encoder route drift")
if champion.get("tag")!="gf_conf_cbg41_e283" or champion.get("public_cases")!=50: raise SystemExit("champion identity drift")
PY

# Produce a classifier-fusion cache. The generic materialized row is only a
# diagnostic; selection below uses component confidence and encoder contours.
export WEAK707_PUBLIC_QUALIFICATION="${oof_terminal}"
export WEAK707_PUBLIC_REQUIRED_DECISION=QUALIFY_FOR_FIXED_PUBLIC
export WEAK707_PUBLIC_OOF_ROOT="${oof}"
export WEAK707_PUBLIC_CALIBRATION="${calibration}"
export WEAK707_PUBLIC_SELECTION="${selection}"
export WEAK707_PUBLIC_CONTOUR_SUMMARY="${contour_summary}"
export WEAK707_PUBLIC_CLASSIFIER_FUSION="${patch_fusion_decision}"
export WEAK707_PUBLIC_BASE="${champion_predictions}"
export WEAK707_PUBLIC_OUTPUT="${classifier_cache}"
export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${classifier_cache}"
export WEAK707_PUBLIC_TAG_PREFIX=cmenc_cache
export WEAK707_PUBLIC_LOCK="${root}/runs/run_component_meta_encoder_classifier_cache_20260825.lock"
bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh" \
  >> "${classifier_cache}/controller.log" 2>&1
[[ -s "${classifier_work}/INFERENCE_COMPLETE.json" && -s "${classifier_work}/AUDIT.json" ]] || exit 3
"${python_bin}" -u "${root}/prepare_component_probability_override.py" \
  --work "${classifier_work}" --component-decision "${component_decision}" \
  --patch-fusion-decision "${patch_fusion_decision}" --output "${override}" \
  > "${output}/component_probability.log" 2>&1

mkdir -p "${encoder_work}"
cp "${classifier_work}/manifest.json" "${encoder_work}/manifest.json"
cp "${classifier_work}/PREPARE_COMPLETE.txt" "${encoder_work}/PREPARE_COMPLETE.txt"
checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
case "${checkpoint_name}" in checkpoint_best.pt|checkpoint_best_classification.pt|checkpoint_best_segmentation.pt|checkpoint_best_cohort_segmentation.pt) ;; *) exit 2 ;; esac
checkpoint_args=(); route_args=(); tta_args=()
for fold in 0 1 2 3 4; do
  checkpoint="${root}/runs/candidate_refiner_v4_cell707_b32_t05_fold${fold}_20260823/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || exit 2
  checkpoint_args+=(--checkpoint "${checkpoint}")
  route_args+=(--candidate-contour-route "b32=${checkpoint}")
  mapfile -t focus_specs < <("${python_bin}" - "${encoder_routes}" "${fold}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); fold=int(sys.argv[2]); rows={int(r["fold"]):r for r in p["fold_records"]}
for name,r in sorted(rows[fold]["routes"].items()):
 if name!="b32": print(f"{name}\t{r['checkpoint']}\t{r['checkpoint_sha256']}")
PY
  )
  for spec in "${focus_specs[@]}"; do
    IFS=$'\t' read -r name path expected_sha <<< "${spec}"
    [[ -s "${path}" && $(sha256sum "${path}" | awk '{print $1}') == "${expected_sha}" ]] || exit 2
    route_args+=(--focus-contour-route "${name}=${path}")
  done
done
mapfile -t source_specs < <("${python_bin}" - "${encoder_routes}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
for source,route in sorted(p["source_to_route"].items()): print(f"{source}={route}")
for route in sorted(set(p["source_to_route"].values())):
 if route.endswith("_tta_xy"): print(f"TTA:{route}")
PY
)
for spec in "${source_specs[@]}"; do
  if [[ "${spec}" == TTA:* ]]; then tta_args+=(--tta-route "${spec#TTA:}")
  else route_args+=(--source-contour-route "${spec}"); fi
done
focus_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["focus_input_mode"])' "${encoder_routes}")
inference_matches() {
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${encoder_work}/INFERENCE_COMPLETE.json" "${checkpoint_args[@]}" \
    "${route_args[@]}" --focus-routed-input-mode "${focus_mode}" > /dev/null 2>&1 || return 1
  "${python_bin}" "${root}/validate_focus_tta_xy_marker.py" --work "${encoder_work}" \
    "${tta_args[@]}" > /dev/null 2>&1
}
gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
wait_any_gpu() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1); inference_gpu=""
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]] && inference_gpu=0
    [[ -z "${inference_gpu}" && "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]] && inference_gpu=1
    [[ -n "${inference_gpu}" ]] && return 0
    printf '%s waiting for public encoder inference GPU: gpu0=%s gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}
if ! inference_matches; then
  wait_any_gpu
  CUDA_VISIBLE_DEVICES="${inference_gpu}" "${python_bin}" -u \
    "${root}/apply_candidate_refiner_focus_tta_xy.py" infer --work "${encoder_work}" \
    "${checkpoint_args[@]}" "${route_args[@]}" --batch-size 4 --workers 2 --resume \
    > "${output}/infer_encoder.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${encoder_work}" --output "${encoder_work}/AUDIT.json" --workers 4 \
  --allow-external-patches --require-predictions > "${output}/audit_encoder.log" 2>&1

"${python_bin}" -u "${root}/export_selected_candidate_refiner_grid.py" \
  --calibration "${calibration}" --tag "${tag}" --output "${grid}" > "${output}/selected_grid.log" 2>&1
read -r segmentation deployment min_seed min_overlap min_ratio max_ratio <<< "$(
  "${python_bin}" - "${calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); s=p["selected"]; seg=float(s["segmentation_threshold"]); deployment=p.get("calibration_deployment")
g=p.get("geometry_gates",{}); values=[float(g.get(k,d)) for k,d in (("min_seed_coverage",0.0),("min_refined_overlap",0.0),("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not 0<seg<1 or deployment not in {"standalone","anchor_currentgf"} or not all(math.isfinite(v) and v>=0 for v in values): raise SystemExit("invalid calibration")
print(f"{seg:.12g}",deployment,*(f"{v:.12g}" for v in values))
PY
)"
empty=(); [[ "${deployment}" == standalone ]] && empty+=(--empty-base)
"${python_bin}" -u "${root}/materialize_candidate_refiner_component_grid.py" \
  --work "${encoder_work}" --base "${champion_predictions}" "${empty[@]}" \
  --output-root "${candidate_root}" --configurations "${grid}" --fold 0 \
  --segmentation-threshold "${segmentation}" --component-probabilities "${override}" \
  --min-seed-coverage "${min_seed}" --min-refined-overlap "${min_overlap}" \
  --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
  --writer-workers 2 > "${output}/materialize.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
  --gt "${labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
"${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
  --anchor "${anchor_metrics}" --candidate "${metrics}" --anchor-name gf_touchcomp_0825 \
  --candidate-name component_meta_encoder_composite --min-dsc-gain 0.003 \
  --min-dsc-win-rate 0.52 --min-nsd-gain 0.0 --min-nsd-uncertainty-bound -0.005 \
  --bootstrap 20000 --seed 20260825 --output "${gate}" > "${output}/gate.log"
dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
gate_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${gate}")
printf 'variant\tdeployment\tDSC\tNSD\tcombined\tmetrics\tgate\n' > "${summary}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${deployment}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${gate_decision}" >> "${summary}"
if [[ "${gate_decision}" == PROMOTE ]]; then
  write_terminal COMPLETE QUALIFY_FOR_HIDDEN "passed fixed 50-case DSC-first/NSD gate versus touchcomp"
else
  write_terminal COMPLETE KEEP_TOUCHCOMP "did not pass fixed 50-case gate versus touchcomp"
fi
trap - EXIT
cat "${summary}"
