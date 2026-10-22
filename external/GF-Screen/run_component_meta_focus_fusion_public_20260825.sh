#!/usr/bin/env bash
set -euo pipefail

# Fixed-public consumer for the exact-707 component meta-classifier. The large
# B32+Focus cache is produced by the audited shared controller, but the final
# probabilities and materialization are component-specific. Hidden data and
# submission packaging are deliberately unreachable from this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
oof="${root}/analysis/component_meta_focus_fusion_fullvolume_oof_20260824"
oof_terminal="${oof}/COMPONENT_META_OOF_TERMINAL.json"
calibration="${oof}/deployment_calibration.json"
component_decision="${root}/analysis/component_meta_classifier_frozen_focus_fusion_20260824/DECISION.json"
patch_fusion_decision="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
contour_summary="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_patch_oof.json"
selection="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/checkpoint_selection.json"
champion="${root}/analysis/DEPLOYMENT_CHAMPION.json"
champion_metrics="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/exact0.metrics.csv"
champion_predictions="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf"
labels="${project}/validation/Validation-Public-Labels"
cache="${root}/public_sweeps/component_meta_focus_fusion_cache_20260825"
work="${cache}/work_bbox24_ss_v4_cell48"
output="${root}/public_sweeps/component_meta_focus_fusion_20260825"
override="${output}/component_probabilities.npz"
grid="${output}/selected_grid.json"
candidate_root="${output}/predictions"
tag="component_meta_selected"
prediction="${candidate_root}/${tag}/fold0"
metrics="${output}/${tag}.metrics.csv"
score_log="${output}/${tag}.score.log"
gate="${output}/dsc_first_vs_gfconf.json"
summary="${output}/summary.tsv"
terminal="${output}/PUBLIC_COMPONENT_TERMINAL.json"
lock="${root}/runs/run_component_meta_focus_fusion_public_20260825.lock"

mkdir -p "${output}" "${cache}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "component-meta fixed-public controller already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED|SKIPPED_OOF)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.${existing_status,,}_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected component-meta public terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${oof_terminal}" \
    "${calibration}" "${component_decision}" "${patch_fusion_decision}" \
    "${work}/AUDIT.json" "${metrics}" "${gate}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
oof,calibration,component,patch,audit,metrics,gate,summary,output=map(Path,sys.argv[4:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path):
 return {"path":str(path.resolve()),"sha256":sha(path)} if path.is_file() else None
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "oof_terminal":asset(oof),"calibration":asset(calibration),
 "component_decision":asset(component),"patch_fusion_decision":asset(patch),
 "inference_audit":asset(audit),"metrics":asset(metrics),"gate":asset(gate),
 "summary":asset(summary),"public_read":metrics.is_file(),"hidden_read":False,
 "submission_created":False,"uploaded":False,
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_CHAMPION \
    "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for exact-707 component-meta OOF terminal\n' "$(date -Ins)"
  sleep 30
done
oof_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != QUALIFY_FOR_FIXED_PUBLIC ]]; then
  write_terminal SKIPPED_OOF KEEP_CHAMPION \
    "exact-707 component-meta decision was ${oof_decision}"
  trap - EXIT
  exit 0
fi
for required in "${calibration}" "${component_decision}" "${patch_fusion_decision}" \
    "${contour_summary}" "${selection}" "${champion}" "${champion_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${champion_predictions}" "${labels}"; do
  [[ -d "${directory}" ]] || { echo "missing required directory: ${directory}" >&2; exit 2; }
done
[[ $(find "${champion_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" - "${oof_terminal}" "${component_decision}" \
  "${patch_fusion_decision}" "${champion}" <<'PY'
import json,sys
oof,component,patch,champion=(json.load(open(path,encoding="utf-8")) for path in sys.argv[1:])
if oof.get("status")!="COMPLETE" or oof.get("decision")!="QUALIFY_FOR_FIXED_PUBLIC":
 raise SystemExit("component OOF is not qualified and complete")
if oof.get("public_read") is not False or oof.get("hidden_read") is not False:
 raise SystemExit("component OOF provenance is contaminated")
if component.get("decision")!="QUALIFY_COMPONENT_META_CLASSIFIER":
 raise SystemExit("component classifier decision drift")
if patch.get("decision")!="QUALIFY_FOCUS_CLASSIFIER_FUSION":
 raise SystemExit("patch fusion decision drift")
if champion.get("tag")!="gf_conf_cbg41_e283" or champion.get("public_cases")!=50:
 raise SystemExit("immutable champion identity drift")
PY

# Reuse the audited generic controller only to create the exact B32+Focus
# inference cache. Its patch-threshold public row is diagnostic and is never
# used for component selection or promotion below.
export WEAK707_PUBLIC_QUALIFICATION="${oof_terminal}"
export WEAK707_PUBLIC_REQUIRED_DECISION=QUALIFY_FOR_FIXED_PUBLIC
export WEAK707_PUBLIC_OOF_ROOT="${oof}"
export WEAK707_PUBLIC_CALIBRATION="${calibration}"
export WEAK707_PUBLIC_SELECTION="${selection}"
export WEAK707_PUBLIC_CONTOUR_SUMMARY="${contour_summary}"
export WEAK707_PUBLIC_CLASSIFIER_FUSION="${patch_fusion_decision}"
export WEAK707_PUBLIC_BASE="${champion_predictions}"
export WEAK707_PUBLIC_OUTPUT="${cache}"
export WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${cache}"
export WEAK707_PUBLIC_TAG_PREFIX=cmcache
export WEAK707_PUBLIC_LOCK="${root}/runs/run_component_meta_focus_fusion_public_cache_20260825.lock"
bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh" \
  >> "${cache}/controller.log" 2>&1
[[ -s "${work}/INFERENCE_COMPLETE.json" && -s "${work}/AUDIT.json" ]] || {
  echo "audited public B32+Focus cache was not produced" >&2; exit 3;
}

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
if not 0<seg<1 or deployment not in {"standalone","anchor_currentgf"}:
 raise SystemExit("invalid frozen component deployment")
g=p.get("geometry_gates",{})
values=[float(g.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not all(math.isfinite(x) and x>=0 for x in values) or values[0]>1 or values[1]>1 or values[2]>values[3]:
 raise SystemExit("invalid frozen component geometry gates")
print(f"{seg:.12g}",deployment,*(f"{x:.12g}" for x in values))
PY
)"
empty_arg=()
[[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
"${python_bin}" -u "${root}/materialize_candidate_refiner_component_grid.py" \
  --work "${work}" --base "${champion_predictions}" "${empty_arg[@]}" \
  --output-root "${candidate_root}" --configurations "${grid}" --fold 0 \
  --segmentation-threshold "${segmentation}" --component-probabilities "${override}" \
  --min-seed-coverage "${min_seed}" --min-refined-overlap "${min_overlap}" \
  --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
  --writer-workers 2 > "${output}/materialize.log" 2>&1
[[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
  > "${score_log}" 2>&1
"${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
  --anchor "${champion_metrics}" --candidate "${metrics}" \
  --anchor-name gf_conf_cbg41_e283 --candidate-name component_meta_focus_fusion \
  --min-dsc-gain 0.015 --min-dsc-win-rate 0.55 --min-nsd-gain 0.0 \
  --min-nsd-uncertainty-bound -0.005 --bootstrap 20000 --seed 20260825 \
  --output "${gate}" > "${output}/gate.log"
dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
gate_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${gate}")
printf 'variant\tdeployment\tDSC\tNSD\tcombined\tmetrics\tgate\n' > "${summary}"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${deployment}" \
  "${dsc}" "${nsd}" "${combined}" "${metrics}" "${gate_decision}" >> "${summary}"
if [[ "${gate_decision}" == PROMOTE ]]; then
  write_terminal COMPLETE QUALIFY_FOR_HIDDEN \
    "passed fixed 50-case +0.015 DSC/NSD-nonregression gate against gf_conf"
else
  write_terminal COMPLETE KEEP_CHAMPION \
    "did not pass fixed 50-case +0.015 DSC/NSD-nonregression gate against gf_conf"
fi
trap - EXIT
cat "${summary}"
