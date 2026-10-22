#!/usr/bin/env bash
set -euo pipefail

# Exact-707 local-only composition of the qualified component classifier with
# encoder-adapted routed contours. Component probabilities are reused from the
# audited frozen classifier-fusion branch, while segmentation probabilities
# come from the encoder composite. No challenge data or archive creation is
# reachable from this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
primary_calibration="${primary}/deployment_calibration.json"
selection="${root}/analysis/candidate_refiner_v4_cell707_training_20260823/checkpoint_selection.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
component_decision="${root}/analysis/component_meta_classifier_frozen_focus_fusion_20260824/DECISION.json"
component_root="${root}/analysis/component_meta_focus_fusion_fullvolume_oof_20260824"
component_terminal="${component_root}/COMPONENT_META_OOF_TERMINAL.json"
component_calibration="${component_root}/deployment_calibration.json"
encoder_root="${root}/analysis/focus_encoder_upgrade_composite_fullvolume_oof_20260824"
encoder_terminal="${encoder_root}/TERMINAL.json"
encoder_metrics="${encoder_root}/composite.metrics.csv"
encoder_routes="${root}/analysis/focus_encoder_upgrade_composite_routes_20260824/DECISION.json"
frozen_root="${root}/analysis/focus_composite_fullvolume_oof_20260824"
frozen_terminal="${frozen_root}/TERMINAL.json"
frozen_metrics="${frozen_root}/composite.metrics.csv"
fusion_root="${root}/analysis/focus_classifier_fusion_frozen_fullvolume_oof_20260824"
fusion_terminal="${fusion_root}/FOCUS_OOF_TERMINAL.json"
fusion_calibration="${fusion_root}/deployment_calibration.json"
output="${root}/analysis/component_meta_encoder_composite_fullvolume_oof_20260825"
summary="${output}/summary.tsv"
calibration="${output}/deployment_calibration.json"
anchor_selection="${output}/anchor_selection.json"
gate="${output}/dsc_first_vs_strongest_qualified_anchor.json"
terminal="${output}/TERMINAL.json"
lock="${root}/runs/run_component_meta_encoder_composite_fullvolume_oof_20260825.lock"
expected_cases=707

mkdir -p "${output}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "component-meta encoder controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

write_terminal() {
  local status="$1" decision="$2" reason="$3"
  "${python_bin}" - "${status}" "${decision}" "${reason}" \
    "${component_decision}" "${component_terminal}" "${encoder_terminal}" \
    "${calibration}" "${anchor_selection}" "${gate}" "${summary}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
status,decision,reason=sys.argv[1:4]
component,component_terminal,encoder_terminal,calibration,anchor,gate,summary,output=map(Path,sys.argv[4:])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":status,"decision":decision,"reason":reason,
 "component_decision":str(component.resolve()),
 "component_decision_sha256":sha(component) if component.is_file() else None,
 "component_terminal":str(component_terminal.resolve()),
 "component_terminal_sha256":sha(component_terminal) if component_terminal.is_file() else None,
 "encoder_terminal":str(encoder_terminal.resolve()),
 "encoder_terminal_sha256":sha(encoder_terminal) if encoder_terminal.is_file() else None,
 "calibration":str(calibration.resolve()) if calibration.is_file() else None,
 "calibration_sha256":sha(calibration) if calibration.is_file() else None,
 "anchor_selection":str(anchor.resolve()) if anchor.is_file() else None,
 "anchor_selection_sha256":sha(anchor) if anchor.is_file() else None,
 "gate":str(gate.resolve()) if gate.is_file() else None,
 "gate_sha256":sha(gate) if gate.is_file() else None,
 "summary":str(summary.resolve()) if summary.is_file() else None,
 "summary_sha256":sha(summary) if summary.is_file() else None,
 "labels_read":True,"public_read":False,"hidden_read":False,
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
  [[ -s "${terminal}" ]] || write_terminal FAILED KEEP_QUALIFIED_ANCHOR \
    "controller failed at line ${line} with status ${status}"
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${encoder_terminal}" || ! -s "${component_terminal}" ]]; do
  printf '%s waiting for encoder-composite and component-meta exact-707 terminals\n' "$(date -Ins)"
  sleep 30
done
encoder_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${encoder_terminal}")
component_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("status",""))' "${component_terminal}")
if [[ "${encoder_status}" != COMPLETE || "${component_status}" != COMPLETE ]]; then
  write_terminal SKIPPED KEEP_QUALIFIED_ANCHOR \
    "required exact-707 branch was not complete"
  trap - EXIT
  exit 0
fi
for required in "${primary_calibration}" "${selection}" "${manifest}" \
    "${component_decision}" "${component_calibration}" "${encoder_metrics}" \
    "${encoder_routes}" "${frozen_terminal}" "${fusion_terminal}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
"${python_bin}" - "${component_decision}" "${component_terminal}" \
  "${encoder_terminal}" "${encoder_routes}" "${manifest}" <<'PY'
import json,sys
component,component_terminal,encoder_terminal,routes,manifest=(json.load(open(p,encoding="utf-8")) for p in sys.argv[1:])
if component.get("decision")!="QUALIFY_COMPONENT_META_CLASSIFIER": raise SystemExit("component classifier not qualified")
if component.get("cases")!=707 or component.get("folds")!=5: raise SystemExit("component coverage mismatch")
if component_terminal.get("status")!="COMPLETE": raise SystemExit("component full-volume branch incomplete")
if encoder_terminal.get("status")!="COMPLETE": raise SystemExit("encoder full-volume branch incomplete")
if routes.get("decision") not in {"QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME","QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_NOTTA_FULL_VOLUME"}:
 raise SystemExit("encoder route identity did not qualify")
items=manifest.get("items",[])
if len(items)!=707 or any(item.get("complete_label") is not True for item in items): raise SystemExit("exact-707 manifest mismatch")
PY

read -r deployment min_seed min_overlap min_ratio max_ratio <<< "$(
  "${python_bin}" - "${primary_calibration}" <<'PY'
import json,math,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); deployment=p.get("calibration_deployment","standalone")
g=p.get("geometry_gates",{}); values=[float(g.get(k,d)) for k,d in (
 ("min_seed_coverage",0.0),("min_refined_overlap",0.0),("min_volume_ratio",0.0),("max_volume_ratio",1e12))]
if deployment not in {"standalone","anchor_currentgf"}: raise SystemExit("invalid deployment")
if not all(math.isfinite(v) and v>=0 for v in values) or values[0]>1 or values[1]>1 or values[2]>values[3]: raise SystemExit("invalid geometry")
print(deployment,*(f"{v:.12g}" for v in values))
PY
)"

# The component probabilities were generated from the qualified Focus+B32
# classifier fusion. Reusing them avoids feeding B32-only confidence values to
# a meta-classifier trained on fused confidence features.
for fold in 0 1 2 3 4; do
  encoder_work="${encoder_root}/fold${fold}/work_bbox24"
  fusion_work="${fusion_root}/fold${fold}/work_bbox24"
  override="${component_root}/fold${fold}.component_probabilities.npz"
  sidecar="${component_root}/fold${fold}.component_probabilities.json"
  for required in "${encoder_work}/INFERENCE_COMPLETE.json" "${encoder_work}/AUDIT.json" \
      "${fusion_work}/manifest.json" "${encoder_work}/manifest.json" "${override}" "${sidecar}"; do
    [[ -s "${required}" ]] || { echo "missing fold ${fold} asset: ${required}" >&2; exit 2; }
  done
  "${python_bin}" - "${encoder_work}/manifest.json" "${fusion_work}/manifest.json" \
    "${override}" "${sidecar}" "${fold}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
encoder,fusion,override,sidecar=map(Path,sys.argv[1:5]); fold=int(sys.argv[5])
ids=lambda p:{str(r["patch_id"]) for r in json.load(open(p,encoding="utf-8")).get("records",[])}
encoder_ids,fusion_ids=ids(encoder),ids(fusion)
if not encoder_ids or encoder_ids!=fusion_ids: raise SystemExit(f"fold {fold} patch identity mismatch")
p=json.load(open(sidecar,encoding="utf-8")); sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if p.get("fold")!=fold or p.get("exact_patch_coverage") is not True or p.get("public_read") is not False or p.get("hidden_read") is not False: raise SystemExit("invalid override sidecar")
if p.get("output_sha256")!=sha(override): raise SystemExit("override hash mismatch")
with np.load(override,allow_pickle=False) as a: observed={str(x) for x in a["patch_id"]}
if observed!=encoder_ids: raise SystemExit("override does not cover encoder patches")
PY
done

grid="${output}/materialization_grid.json"
"${python_bin}" - "${component_decision}" "${encoder_routes}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
component=json.load(open(sys.argv[1],encoding="utf-8")); routes=json.load(open(sys.argv[2],encoding="utf-8")); output=Path(sys.argv[3])
thresholds=[float(v) for v in component["deployment_thresholds"]]
if thresholds!=[0.54,0.59,0.64]: raise SystemExit("component threshold identity drift")
source_seg={str(k):float(v) for k,v in routes.get("source_segmentation_thresholds",{}).items()}
expected={"epoch105","currentgf","autopet_weak","lndb_weak",*(f"lung_cell{i}" for i in range(8))}
if set(source_seg)!=expected: raise SystemExit("encoder contour threshold coverage mismatch")
groups=component["by_source_group"]
source_cls={
 "epoch105":float(groups["mature"]["candidate"]["threshold"]),
 "currentgf":float(groups["mature"]["candidate"]["threshold"]),
 "autopet_weak":float(groups["weak_sources"]["candidate"]["threshold"]),
 "lndb_weak":float(groups["weak_sources"]["candidate"]["threshold"]),
 **{f"lung_cell{i}":float(groups["lung_cells"]["candidate"]["threshold"]) for i in range(8)},
}
configs=[]
for threshold in thresholds:
 tag=f"enc_component_c{threshold:.2f}_srcseg".replace(".","p")
 configs.append({"tag":tag,"classification_threshold":threshold,"mode":"refined","source_classification_thresholds":{},"source_segmentation_thresholds":source_seg})
configs.append({"tag":"enc_component_srcboth_c0p59","classification_threshold":0.59,"mode":"refined","source_classification_thresholds":source_cls,"source_segmentation_thresholds":source_seg})
output.write_text(json.dumps({"schema_version":1,"selection_basis":"frozen exact707 component probabilities plus encoder-qualified contour thresholds","configurations":configs},indent=2)+"\n",encoding="utf-8")
PY

materialize_fold() {
  local fold="$1" work="${encoder_root}/fold${fold}/work_bbox24" base empty=()
  if [[ "${deployment}" == anchor_currentgf ]]; then base="${primary}/staging/fold${fold}/proposals/currentgf"
  else base="${primary}/staging/fold${fold}/proposals/epoch105"; empty+=(--empty-base); fi
  "${python_bin}" -u "${root}/materialize_candidate_refiner_component_grid.py" \
    --work "${work}" --base "${base}" "${empty[@]}" --output-root "${output}" \
    --configurations "${grid}" --fold "${fold}" --segmentation-threshold 0.5 \
    --component-probabilities "${component_root}/fold${fold}.component_probabilities.npz" \
    --min-seed-coverage "${min_seed}" --min-refined-overlap "${min_overlap}" \
    --min-volume-ratio "${min_ratio}" --max-volume-ratio "${max_ratio}" \
    --writer-workers 2 > "${output}/fold${fold}.materialize.log" 2>&1
}
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do materialize_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tDSC\tNSD\tcombined\tmetrics\n' > "${summary}"
mapfile -t rows < <("${python_bin}" - "${grid}" <<'PY'
import json,sys
for r in json.load(open(sys.argv[1],encoding="utf-8"))["configurations"]:
 print("\t".join((r["tag"],str(r["classification_threshold"]),r["mode"],json.dumps(r.get("source_classification_thresholds",{}),sort_keys=True,separators=(",",":")),json.dumps(r.get("source_segmentation_thresholds",{}),sort_keys=True,separators=(",",":")))))
PY
)
for record in "${rows[@]}"; do
  IFS=$'\t' read -r tag classification mode source_cls source_seg <<< "${record}"
  combined="${output}/${tag}/combined"; mkdir -p "${combined}"
  "${python_bin}" - "${output}/${tag}" "${combined}" "${primary}/staging/all/labels" "${expected_cases}" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:4]); expected=int(sys.argv[4]); seen={}
for fold in range(5):
 for source in (root/f"fold{fold}").glob("*.nii.gz"):
  if source.name in seen: raise SystemExit(f"duplicate OOF case: {source.name}")
  target=combined/source.name
  if not target.exists(): target.symlink_to(source.resolve())
  seen[source.name]=source
truth={p.name for p in labels.glob("*.nii.gz")}
if len(truth)!=expected or set(seen)!=truth: raise SystemExit("exact-707 OOF coverage mismatch")
PY
  metrics="${output}/${tag}.metrics.csv"; score="${output}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${combined}" \
    --gt "${primary}/staging/all/labels" --csv "${metrics}" --workers 4 > "${score}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score}")
  combined_score=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t0.5\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${tag}" "${classification}" "${mode}" "${source_cls}" "${source_seg}" \
    "${dsc}" "${nsd}" "${combined_score}" "${metrics}" >> "${summary}"
done
{ head -1 "${summary}"; tail -n +2 "${summary}" | sort -t $'\t' -k7,7gr -k8,8gr; } > "${output}/summary_by_dsc.tsv"

"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 \
  --seed 20260825 > "${output}/summarize.log" 2>&1
"${python_bin}" - "${calibration}" "${deployment}" "${min_seed}" "${min_overlap}" "${min_ratio}" "${max_ratio}" <<'PY'
import json,os,sys
from pathlib import Path
path=Path(sys.argv[1]); p=json.load(open(path,encoding="utf-8")); names=("min_seed_coverage","min_refined_overlap","min_volume_ratio","max_volume_ratio")
gates=dict(zip(names,map(float,sys.argv[3:]))); p["calibration_deployment"]=sys.argv[2]; p["geometry_gates"]=gates
p["selection_basis"] += "; component confidences and encoder contour thresholds fixed before full-volume scoring"
p["selected"]["calibration_deployment"]=sys.argv[2]; p["selected"]["geometry_gates"]=gates
temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,path)
PY

# Bind the comparison to the strongest exact-707 branch that already passed
# its own promotion gate. Raw but rejected branches cannot become anchors.
anchor_metrics=$("${python_bin}" - "${primary_calibration}" "${frozen_terminal}" "${frozen_metrics}" \
  "${fusion_terminal}" "${fusion_calibration}" "${encoder_terminal}" "${encoder_metrics}" \
  "${component_terminal}" "${component_calibration}" "${anchor_selection}" <<'PY'
import csv,datetime,hashlib,json,os,sys
from pathlib import Path
primary,ft,fm,ut,uc,et,em,ct,cc,out=map(Path,sys.argv[1:]); candidates=[]
def mean(path,key):
 with path.open(newline="",encoding="utf-8") as h: values=[float(r[key]) for r in csv.DictReader(h)]
 return sum(values)/len(values)
def add(name,path): candidates.append({"name":name,"metrics":str(path.resolve()),"Lesion_DSC":mean(path,"Lesion_DSC"),"Lesion_NSD":mean(path,"Lesion_NSD")})
p=json.load(open(primary,encoding="utf-8")); add("primary_exact707",Path(p["selected"]["metrics"]))
for name,terminal,metrics,required in (("frozen_composite",ft,fm,"PROMOTE"),("encoder_composite",et,em,"PROMOTE")):
 t=json.load(open(terminal,encoding="utf-8"));
 if t.get("status")=="COMPLETE" and t.get("decision")==required and metrics.is_file(): add(name,metrics)
for name,terminal,cal,required in (("frozen_classifier_fusion",ut,uc,"PROMOTE"),("component_meta_fusion",ct,cc,"QUALIFY_FOR_FIXED_PUBLIC")):
 t=json.load(open(terminal,encoding="utf-8"));
 if t.get("status")=="COMPLETE" and t.get("decision")==required and cal.is_file():
  q=json.load(open(cal,encoding="utf-8")); add(name,Path(q["selected"]["metrics"]))
selected=max(candidates,key=lambda r:(r["Lesion_DSC"],r["Lesion_NSD"]))
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"policy":"highest DSC among branches that passed their own exact-707 gate","candidates":candidates,"selected":selected,"public_read":False,"hidden_read":False}
tmp=out.with_name(f".{out.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,out)
print(selected["metrics"])
PY
)
candidate_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected"]["metrics"])' "${calibration}")
"${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name strongest_prequalified_exact707 --candidate-name component_meta_encoder_composite \
  --min-dsc-gain 0.003 --min-dsc-win-rate 0.52 --min-nsd-gain 0.0 \
  --min-nsd-uncertainty-bound -0.002 --bootstrap 20000 --seed 20260825 \
  --output "${gate}" > "${output}/gate.log"
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${gate}")
if [[ "${decision}" == PROMOTE ]]; then
  write_terminal COMPLETE QUALIFY_FOR_FIXED_PUBLIC \
    "component-selected encoder contours passed exact-707 DSC-first/NSD-nonregression gate"
else
  write_terminal COMPLETE KEEP_QUALIFIED_ANCHOR \
    "component-selected encoder contours did not pass the exact-707 gate"
fi
trap - EXIT
cat "${output}/summary_by_dsc.tsv"
