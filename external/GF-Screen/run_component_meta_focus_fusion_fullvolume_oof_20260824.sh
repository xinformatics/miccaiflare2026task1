#!/usr/bin/env bash
set -euo pipefail

# Exact-707 deployment test for the cross-fitted component meta-classifier.
# The controller reuses frozen B32+Focus inference caches, applies fold-specific
# component models, and reads only complete internal labels. Public/hidden
# inputs and submission packaging are deliberately unreachable from this file.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
focus="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
focus_terminal="${focus}/FOCUS_OOF_TERMINAL.json"
focus_calibration="${focus}/deployment_calibration.json"
component_decision="${root}/analysis/component_meta_classifier_frozen_focus_fusion_20260824/DECISION.json"
patch_fusion_decision="${root}/analysis/focus_classifier_fusion_frozen_20260824/DECISION.json"
selection="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/checkpoint_selection.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
output="${root}/analysis/component_meta_focus_fusion_fullvolume_oof_20260824"
summary="${output}/summary.tsv"
calibration="${output}/deployment_calibration.json"
gate="${output}/dsc_first_vs_strongest_anchor.json"
terminal="${output}/COMPONENT_META_OOF_TERMINAL.json"
lock="${root}/runs/run_component_meta_focus_fusion_fullvolume_oof_20260824.lock"
expected_cases=707

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "component-meta full-volume controller already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE) exit 0 ;;
    FAILED|SKIPPED)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.${existing_status,,}_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected component-meta OOF terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${component_decision}" \
    "${patch_fusion_decision}" "${focus_terminal}" "${calibration}" "${gate}" \
    "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
component,patch,focus,calibration,gate,summary,output=map(Path,sys.argv[4:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "component_decision":str(component.resolve()),
 "component_decision_sha256":sha(component) if component.is_file() else None,
 "patch_fusion_decision":str(patch.resolve()),
 "patch_fusion_decision_sha256":sha(patch) if patch.is_file() else None,
 "focus_terminal":str(focus.resolve()),
 "focus_terminal_sha256":sha(focus) if focus.is_file() else None,
 "calibration":str(calibration.resolve()) if calibration.is_file() else None,
 "calibration_sha256":sha(calibration) if calibration.is_file() else None,
 "gate":str(gate.resolve()) if gate.is_file() else None,
 "gate_sha256":sha(gate) if gate.is_file() else None,
 "summary":str(summary.resolve()) if summary.is_file() else None,
 "summary_sha256":sha(summary) if summary.is_file() else None,
 "public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False,
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_ANCHOR \
    "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${focus_terminal}" ]]; do
  printf '%s waiting for frozen Focus-fusion full-volume terminal\n' "$(date -Ins)"
  sleep 30
done
focus_status=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' \
  "${focus_terminal}")
if [[ "${focus_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_ANCHOR "Focus-fusion full-volume status was ${focus_status}"
  trap - EXIT
  exit 0
fi
for required in "${focus_calibration}" "${component_decision}" \
    "${patch_fusion_decision}" "${selection}" "${manifest}" \
    "${primary}/deployment_calibration.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
"${python_bin}" - "${component_decision}" "${patch_fusion_decision}" \
  "${focus_terminal}" "${manifest}" <<'PY'
import json,sys
component,patch,focus,manifest=map(lambda p:json.load(open(p,encoding="utf-8")),sys.argv[1:])
if component.get("decision")!="QUALIFY_COMPONENT_META_CLASSIFIER":
 raise SystemExit("component classifier did not qualify")
if component.get("cases")!=707 or component.get("folds")!=5:
 raise SystemExit("component classifier coverage mismatch")
if patch.get("decision")!="QUALIFY_FOCUS_CLASSIFIER_FUSION":
 raise SystemExit("patch fusion did not qualify")
if focus.get("status")!="COMPLETE": raise SystemExit("Focus terminal is incomplete")
items=manifest.get("items",[])
if len(items)!=707 or any(item.get("complete_label") is not True for item in items):
 raise SystemExit("exact-707 complete-label manifest mismatch")
PY

read -r segmentation deployment min_seed min_overlap min_ratio max_ratio <<< "$(
  "${python_bin}" - "${focus_calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); selected=p["selected"]
seg=float(selected["segmentation_threshold"]); deployment=p.get("calibration_deployment")
if not 0<seg<1 or deployment not in {"standalone","anchor_currentgf"}:
 raise SystemExit("invalid frozen Focus deployment")
g=p.get("geometry_gates",{})
values=[float(g.get(name,default)) for name,default in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),
 ("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if not all(math.isfinite(x) and x>=0 for x in values) or values[0]>1 or values[1]>1 or values[2]>values[3]:
 raise SystemExit("invalid frozen geometry gates")
print(f"{seg:.12g}",deployment,*(f"{x:.12g}" for x in values))
PY
)"

# Prepare hash-bound fold-specific component probabilities. Two-at-a-time keeps
# random reads over the large NPZ cache from saturating memory and storage.
prepare_fold() {
  local fold="$1" work="${focus}/fold${fold}/work_bbox24"
  local override="${output}/fold${fold}.component_probabilities.npz"
  [[ -s "${work}/INFERENCE_COMPLETE.json" && -s "${work}/AUDIT.json" ]] || {
    echo "missing audited fused cache for fold ${fold}" >&2; return 2;
  }
  if [[ ! -s "${override}" || ! -s "${override%.npz}.json" ]]; then
    "${python_bin}" -u "${root}/prepare_component_probability_override.py" \
      --work "${work}" --component-decision "${component_decision}" \
      --patch-fusion-decision "${patch_fusion_decision}" --fold "${fold}" \
      --output "${override}" > "${output}/fold${fold}.component_probability.log" 2>&1
  fi
}
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do prepare_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

grid="${output}/materialization_grid.json"
"${python_bin}" - "${component_decision}" "${focus_calibration}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
component=json.load(open(sys.argv[1],encoding="utf-8"))
focus=json.load(open(sys.argv[2],encoding="utf-8")); output=Path(sys.argv[3])
thresholds=[float(value) for value in component["deployment_thresholds"]]
if thresholds!=[0.54,0.59,0.64]: raise SystemExit("component threshold identity drift")
selected=focus["selected"]
source_seg={str(k):float(v) for k,v in selected.get("source_segmentation_thresholds",{}).items()}
groups=component["by_source_group"]
source_cls={
 "epoch105":float(groups["mature"]["candidate"]["threshold"]),
 "currentgf":float(groups["mature"]["candidate"]["threshold"]),
 "autopet_weak":float(groups["weak_sources"]["candidate"]["threshold"]),
 "lndb_weak":float(groups["weak_sources"]["candidate"]["threshold"]),
 **{f"lung_cell{i}":float(groups["lung_cells"]["candidate"]["threshold"]) for i in range(8)},
}
seg=float(selected["segmentation_threshold"]); configs=[]
for threshold in thresholds:
 tag=f"component_c{threshold:.2f}_s{seg:.2f}_refined".replace(".","p")
 configs.append({"tag":tag,"classification_threshold":threshold,"mode":"refined",
  "source_classification_thresholds":{},"source_segmentation_thresholds":{}})
configs.append({"tag":f"component_src_c0p59_s{seg:.2f}_refined".replace(".","p"),
 "classification_threshold":0.59,"mode":"refined",
 "source_classification_thresholds":source_cls,"source_segmentation_thresholds":{}})
if source_seg:
 configs.append({"tag":f"component_srcseg_c0p59_s{seg:.2f}_refined".replace(".","p"),
  "classification_threshold":0.59,"mode":"refined",
  "source_classification_thresholds":{},"source_segmentation_thresholds":source_seg})
 configs.append({"tag":f"component_srcboth_c0p59_s{seg:.2f}_refined".replace(".","p"),
  "classification_threshold":0.59,"mode":"refined",
  "source_classification_thresholds":source_cls,"source_segmentation_thresholds":source_seg})
output.write_text(json.dumps({"schema_version":1,"selection_basis":"frozen exact707 component OOF thresholds and frozen Focus contour calibration","configurations":configs},indent=2)+"\n",encoding="utf-8")
PY

materialize_fold() {
  local fold="$1" work="${focus}/fold${fold}/work_bbox24" base empty=()
  if [[ "${deployment}" == anchor_currentgf ]]; then
    base="${primary}/staging/fold${fold}/proposals/currentgf"
  else
    base="${primary}/staging/fold${fold}/proposals/epoch105"
    empty+=(--empty-base)
  fi
  "${python_bin}" -u "${root}/materialize_candidate_refiner_component_grid.py" \
    --work "${work}" --base "${base}" "${empty[@]}" --output-root "${output}" \
    --configurations "${grid}" --fold "${fold}" --segmentation-threshold "${segmentation}" \
    --component-probabilities "${output}/fold${fold}.component_probabilities.npz" \
    --min-seed-coverage "${min_seed}" --min-refined-overlap "${min_overlap}" \
    --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
    --writer-workers 2 > "${output}/fold${fold}.materialize.log" 2>&1
}
# Four or more configurations retain multiple output masks per case. A 2+2+1
# schedule stays below the proven full-volume memory envelope.
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do materialize_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tDSC\tNSD\tcombined\tmetrics\n' > "${summary}"
mapfile -t config_rows < <("${python_bin}" - "${grid}" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1],encoding="utf-8"))["configurations"]:
 print("\t".join((row["tag"],str(row["classification_threshold"]),row["mode"],
  json.dumps(row.get("source_classification_thresholds",{}),sort_keys=True,separators=(",",":")),
  json.dumps(row.get("source_segmentation_thresholds",{}),sort_keys=True,separators=(",",":")))))
PY
)
for record in "${config_rows[@]}"; do
  IFS=$'\t' read -r tag classification mode source_cls source_seg <<< "${record}"
  combined_dir="${output}/${tag}/combined"; mkdir -p "${combined_dir}"
  "${python_bin}" - "${output}/${tag}" "${combined_dir}" \
    "${primary}/staging/all/labels" "${expected_cases}" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:4]); expected=int(sys.argv[4]); observed={}
for fold in range(5):
 for source in (root/f"fold{fold}").glob("*.nii.gz"):
  if source.name in observed: raise SystemExit(f"duplicate OOF case: {source.name}")
  target=combined/source.name
  if target.is_symlink() or target.exists():
   if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit("OOF link collision")
  else: target.symlink_to(source.resolve())
  observed[source.name]=source
expected_names={path.name for path in labels.glob("*.nii.gz")}
if len(expected_names)!=expected or set(observed)!=expected_names: raise SystemExit("exact OOF coverage mismatch")
PY
  metrics="${output}/${tag}.metrics.csv"; score_log="${output}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined_dir}" --gt "${primary}/staging/all/labels" \
    --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${segmentation}" "${mode}" "${source_cls}" \
    "${source_seg}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${summary}"
done
{ head -1 "${summary}"; tail -n +2 "${summary}" | sort -t $'\t' -k7,7gr -k8,8gr; } \
  > "${output}/summary_by_dsc.tsv"

"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 \
  --seed 20260824 > "${output}/summarize.log" 2>&1
"${python_bin}" - "${calibration}" "${focus_calibration}" "${deployment}" \
  "${min_seed}" "${min_overlap}" "${min_ratio}" "${max_ratio}" <<'PY'
import json,os,sys
from pathlib import Path
output,source=map(Path,sys.argv[1:3]); deployment=sys.argv[3]
names=("min_seed_coverage","min_refined_overlap","min_volume_ratio","max_volume_ratio")
gates=dict(zip(names,map(float,sys.argv[4:])))
p=json.load(open(output,encoding="utf-8")); p["geometry_gates"]=gates
p["calibration_deployment"]=deployment
p["fixed_geometry_source"]=str(source.resolve()); p["fixed_deployment_source"]=str(source.resolve())
p["selection_basis"] += "; geometry/deployment fixed from frozen Focus-fusion exact OOF"
p["selected"]["geometry_gates"]=gates; p["selected"]["calibration_deployment"]=deployment
for row in p["all_variants"]:
 row["geometry_gates"]=gates; row["calibration_deployment"]=deployment
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY

anchor_metrics=$("${python_bin}" - "${primary}/deployment_calibration.json" \
  "${focus_calibration}" <<'PY'
import json,sys
rows=[]
for path in sys.argv[1:]:
 p=json.load(open(path,encoding="utf-8")); rows.append((float(p["selected"]["Lesion_DSC"]),p["selected"]["metrics"]))
print(max(rows,key=lambda row:row[0])[1])
PY
)
candidate_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["metrics"])' \
  "${calibration}")
"${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name strongest_exact707_b32_or_focus --candidate-name component_meta_focus_fusion \
  --min-dsc-gain 0.010 --min-dsc-win-rate 0.55 --min-nsd-gain 0.0 \
  --min-nsd-uncertainty-bound -0.005 --bootstrap 20000 --seed 20260824 \
  --output "${gate}" > "${output}/gate.log"
decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${gate}")
if [[ "${decision}" == PROMOTE ]]; then
  write_terminal COMPLETE QUALIFY_FOR_FIXED_PUBLIC \
    "component meta-classifier passed exact-707 DSC-first/NSD-nonregression gate"
else
  write_terminal COMPLETE KEEP_ANCHOR \
    "component meta-classifier did not pass exact-707 DSC-first/NSD-nonregression gate"
fi
trap - EXIT
cat "${output}/summary_by_dsc.tsv"
