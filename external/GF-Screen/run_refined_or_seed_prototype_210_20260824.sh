#!/usr/bin/env bash
set -euo pipefail

# CPU-only feasibility test for component-level selective refinement. It uses
# the already completed 210-case grouped OOF cache, never public/hidden data,
# and cannot package or upload a submission. The exact707 chain is amended only
# if this older-cache control reproduces and a geometry gate improves both DSC
# and NSD.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
source_root="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
output="${root}/analysis/refined_or_seed_prototype_210_20260824"
config="${output}/configurations.json"
complete="${output}/PROTOTYPE_COMPLETE.json"

mkdir -p "${output}"
exec 9>"${root}/runs/refined_or_seed_prototype_210_20260824.lock"
flock -n 9 || { echo "refined-or-seed prototype already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
for fold in 0 1 2 3 4; do
  [[ -s "${source_root}/fold${fold}/work_bbox24/manifest.json" ]] || exit 2
done
[[ $(find "${source_root}/staging/all/labels" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]] || exit 2

"${python_bin}" - "${config}" <<'PY'
import json,sys
from pathlib import Path
output=Path(sys.argv[1])
points=(
 ("control_refined","refined",0.0,0.0,0.0,1e12),
 ("ros_loose","refined_or_seed",0.01,0.01,0.05,20.0),
 ("ros_default","refined_or_seed",0.05,0.05,0.25,4.0),
 ("ros_overlap","refined_or_seed",0.10,0.10,0.25,4.0),
 ("ros_ratio","refined_or_seed",0.0,0.0,0.50,2.0),
 ("ros_consensus","refined_or_seed",0.25,0.25,0.50,2.0),
)
configs=[]
for tag,mode,coverage,overlap,minimum,maximum in points:
 configs.append({
  "tag":tag,"classification_threshold":0.85,"mode":mode,
  "source_classification_thresholds":{},
  "source_segmentation_thresholds":{},
  "source_deployment_modes":{},
  "geometry_gates":{
   "min_seed_coverage":coverage,"min_refined_overlap":overlap,
   "min_volume_ratio":minimum,"max_volume_ratio":maximum,
  },
 })
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n")
PY

materialize_fold() {
  local fold="$1"
  "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
    --work "${source_root}/fold${fold}/work_bbox24" \
    --base "${source_root}/staging/fold${fold}/proposals" \
    --empty-base --output-root "${output}" --configurations "${config}" \
    --fold "${fold}" --segmentation-threshold 0.80 \
    --min-seed-coverage 0 --min-refined-overlap 0 \
    --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
    > "${output}/fold${fold}.materialize.log" 2>&1
}
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do materialize_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

mapfile -t tags < <("${python_bin}" - "${config}" <<'PY'
import json,sys
for row in json.load(open(sys.argv[1]))["configurations"]: print(row["tag"])
PY
)
for tag in "${tags[@]}"; do
  combined="${output}/${tag}/combined"
  mkdir -p "${combined}"
  for fold in 0 1 2 3 4; do
    for source in "${output}/${tag}/fold${fold}"/*.nii.gz; do
      target="${combined}/$(basename "${source}")"
      [[ ! -e "${target}" && ! -L "${target}" ]] || { echo "duplicate ${target}" >&2; exit 3; }
      ln -s "$(realpath "${source}")" "${target}"
    done
  done
  [[ $(find "${combined}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]] || exit 3
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined}" --gt "${source_root}/staging/all/labels" \
    --csv "${output}/${tag}.metrics.csv" --workers 4 \
    > "${output}/${tag}.score.log" 2>&1
done

"${python_bin}" - "${source_root}" "${output}" "${complete}" "${tags[@]}" <<'PY'
import csv,datetime,json,os,sys
from pathlib import Path
source,root,complete=map(Path,sys.argv[1:4]); tags=sys.argv[4:]
def metrics(path):
 with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
 return rows,sum(float(r["Lesion_DSC"]) for r in rows)/len(rows),sum(float(r["Lesion_NSD"]) for r in rows)/len(rows)
anchor_rows,anchor_dsc,anchor_nsd=metrics(source/"stand_c0p85_s0p80_refined.metrics.csv")
control_rows,control_dsc,control_nsd=metrics(root/"control_refined.metrics.csv")
anchor={(r["Name"],r["Lesion_DSC"],r["Lesion_NSD"]) for r in anchor_rows}
control={(r["Name"],r["Lesion_DSC"],r["Lesion_NSD"]) for r in control_rows}
if anchor!=control: raise SystemExit("control does not exactly reproduce prior OOF rows")
rows=[]
for tag in tags:
 _,dsc,nsd=metrics(root/f"{tag}.metrics.csv")
 markers=[]
 for fold in range(5):
  marker=json.load(open(root/tag/f"fold{fold}"/"CANDIDATE_REFINER_COMPLETE.json"))
  markers.append({"fold":fold,"candidates":marker["candidates"],"accepted":marker["accepted"],"gates":marker["gates"]})
 rows.append({"tag":tag,"DSC":dsc,"NSD":nsd,"DSC_gain":dsc-anchor_dsc,"NSD_gain":nsd-anchor_nsd,"folds":markers})
best=max(rows,key=lambda r:(min(r["DSC_gain"],r["NSD_gain"]),r["DSC_gain"],r["NSD_gain"]))
qualifies=best["tag"]!="control_refined" and best["DSC_gain"]>=0.002 and best["NSD_gain"]>=0.002
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "selection_basis":"completed 210-case case-grouped OOF feasibility cache",
 "control_exact":True,"anchor":{"DSC":anchor_dsc,"NSD":anchor_nsd},"rows":rows,
 "best":best,"decision":"BUILD_EXACT707_GEOMETRY_OOF" if qualifies else "REJECT_GEOMETRY_FALLBACK",
 "minimum_gain_each_metric":0.002,"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=complete.with_name(f".{complete.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(temporary,complete)
print(json.dumps(payload,indent=2))
PY
