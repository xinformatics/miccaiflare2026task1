#!/usr/bin/env bash
set -euo pipefail

# Exact-707, case-held-out comparison of a non-destructive additive deployment
# and conservative learned removal of current-GF base components. This reuses
# the primary five-fold probability cache: it performs no training, GPU
# inference, public evaluation, hidden inference, packaging, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
primary_complete="${primary}/FULL_VOLUME_OOF_COMPLETE.json"
primary_calibration="${primary}/deployment_calibration.json"
staging="${primary}/staging"
output="${root}/analysis/candidate_refiner_v4_prune_composition_oof_20260824"
grid="${output}/materialization_grid.json"
summary="${output}/summary.tsv"
calibration="${output}/deployment_calibration.json"
base_robust="${output}/prune_vs_currentgf.robust.json"
additive_robust="${output}/prune_vs_additive.robust.json"
terminal="${output}/FULL_VOLUME_OOF_TERMINAL.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v4_prune_composition_oof_20260824.lock"
flock -n 9 || { echo "prune composition OOF already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${primary_complete}" ]]; do
  printf '%s waiting for primary exact-707 full-volume OOF cache\n' "$(date -Ins)"
  sleep 30
done
for required in "${primary_calibration}" "${staging}/STAGING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${primary_complete}" "${staging}/STAGING_COMPLETE.json" <<'PY'
import json,sys
from pathlib import Path
primary,staging=map(Path,sys.argv[1:])
p=json.loads(primary.read_text(encoding="utf-8")); s=json.loads(staging.read_text(encoding="utf-8"))
if int(p.get("cases",0))!=707 or int(s.get("cases",0))!=707:
    raise SystemExit("exact-707 coverage prerequisite failed")
if p.get("public_read") is not False or p.get("hidden_read") is not False:
    raise SystemExit("primary marker is not label-isolated")
if s.get("proposal_layout")!="named_sources" or not s.get("exact_proposal_coverage"):
    raise SystemExit("source-separated staging audit failed")
PY

# Freeze both variants from the already selected primary operating point. The
# only difference is currentgf=prune_base; all other source thresholds, contour
# thresholds, and source modes remain identical.
"${python_bin}" - "${primary_calibration}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
source=Path(sys.argv[1]); output=Path(sys.argv[2]); c=json.loads(source.read_text(encoding="utf-8"))
required=("classification_threshold","segmentation_threshold")
if any(name not in c for name in required): raise SystemExit("incomplete primary calibration")
common={
 "classification_threshold":float(c["classification_threshold"]),
 "mode":str(c.get("deployment_mode","refined")),
 "source_classification_thresholds":dict(c.get("source_classification_thresholds",{})),
 "source_segmentation_thresholds":dict(c.get("source_segmentation_thresholds",{})),
}
modes=dict(c.get("source_deployment_modes",{}))
additive=dict(common,tag="pc707_additive",source_deployment_modes=modes)
prune_modes=dict(modes); prune_modes["currentgf"]="prune_base"
prune=dict(common,tag="pc707_prune",source_deployment_modes=prune_modes)
payload={"schema_version":1,"primary_calibration":str(source.resolve()),
         "segmentation_threshold":float(c["segmentation_threshold"]),
         "configurations":[additive,prune]}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
segmentation=$("${python_bin}" -c \
  'import json,sys;print(float(json.load(open(sys.argv[1]))["segmentation_threshold"]))' \
  "${primary_calibration}")

materialize_fold() {
  local fold="$1"
  local work="${primary}/fold${fold}/work_bbox24"
  local base="${staging}/fold${fold}/proposals/currentgf"
  for required in "${work}/PREPARE_COMPLETE.txt" "${work}/INFERENCE_COMPLETE.json" "${base}"; do
    [[ -e "${required}" ]] || { echo "missing cached fold asset: ${required}" >&2; return 2; }
  done
  "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
    --work "${work}" --base "${base}" --output-root "${output}" \
    --configurations "${grid}" --fold "${fold}" \
    --segmentation-threshold "${segmentation}" \
    --min-seed-coverage 0 --min-refined-overlap 0 --min-volume-ratio 0 \
    --max-volume-ratio 1000000000000 \
    > "${output}/fold${fold}.materialize.log" 2>&1
}

# Bound RAM while still exploiting independent folds. No GPU is used here.
for group in "0 1" "2 3" "4"; do
  pids=(); status=0
  for fold in ${group}; do materialize_fold "${fold}" & pids+=("$!"); done
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || exit "${status}"
done

combine_folds() {
  local tag="$1" combined="${output}/${tag}/combined"
  mkdir -p "${combined}"
  "${python_bin}" - "${output}/${tag}" "${combined}" "${staging}/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); seen={}
for fold in range(5):
    for source in sorted((root/f"fold{fold}").glob("*.nii.gz")):
        if source.name in seen: raise SystemExit(f"duplicate OOF case: {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source.resolve(strict=True))
        seen[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=707 or set(seen)!=expected: raise SystemExit("OOF coverage mismatch")
PY
}

combine_currentgf() {
  local combined="${output}/pc707_currentgf/combined"
  mkdir -p "${combined}"
  "${python_bin}" - "${staging}" "${combined}" "${staging}/all/labels" <<'PY'
import sys
from pathlib import Path
staging,combined,labels=map(Path,sys.argv[1:]); seen={}
for fold in range(5):
    for source in sorted((staging/f"fold{fold}"/"proposals"/"currentgf").glob("*.nii.gz")):
        if source.name in seen: raise SystemExit(f"duplicate OOF case: {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source.resolve(strict=True))
        seen[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=707 or set(seen)!=expected: raise SystemExit("current-GF coverage mismatch")
PY
}

score() {
  local tag="$1" pred="${output}/${tag}/combined"
  local metrics="${output}/${tag}.metrics.csv" log="${output}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${pred}" --gt "${staging}/all/labels" --csv "${metrics}" --workers 6 \
    > "${log}" 2>&1
}

combine_currentgf
combine_folds pc707_additive
combine_folds pc707_prune
for tag in pc707_currentgf pc707_additive pc707_prune; do score "${tag}"; done

printf 'variant\tDSC\tNSD\tcombined\tmetrics\n' > "${summary}"
for tag in pc707_currentgf pc707_additive pc707_prune; do
  log="${output}/${tag}.score.log"; metrics="${output}/${tag}.metrics.csv"
  dsc=$(awk '/DSC  mean/ {print $3}' "${log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\n' "${tag}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${summary}"
done

"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${output}/pc707_currentgf.metrics.csv" \
  --candidate "${output}/pc707_prune.metrics.csv" \
  --anchor-name currentgf_exact707 --candidate-name pc707_prune \
  --min-dsc-gain 0.002 --min-nsd-gain 0.002 --min-win-rate 0.51 \
  --max-catastrophic-rate 0.05 --bootstrap 20000 --seed 20260824 \
  --output "${base_robust}" > "${output}/prune_vs_currentgf.robust.log"
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${output}/pc707_additive.metrics.csv" \
  --candidate "${output}/pc707_prune.metrics.csv" \
  --anchor-name pc707_additive --candidate-name pc707_prune \
  --min-dsc-gain 0.002 --min-nsd-gain 0.002 --min-win-rate 0.51 \
  --max-catastrophic-rate 0.05 --bootstrap 20000 --seed 20260825 \
  --output "${additive_robust}" > "${output}/prune_vs_additive.robust.log"

"${python_bin}" - "${primary_calibration}" "${grid}" "${summary}" \
  "${base_robust}" "${additive_robust}" "${calibration}" "${terminal}" <<'PY'
import csv,datetime,json,os,sys
from pathlib import Path
primary_path,grid_path,summary_path,base_path,add_path,cal_path,terminal_path=map(Path,sys.argv[1:])
primary=json.loads(primary_path.read_text(encoding="utf-8")); grid=json.loads(grid_path.read_text(encoding="utf-8"))
base=json.loads(base_path.read_text(encoding="utf-8")); additive=json.loads(add_path.read_text(encoding="utf-8"))
with summary_path.open(newline="",encoding="utf-8") as stream:
    rows={row["variant"]:row for row in csv.DictReader(stream,delimiter="\t")}
row=rows["pc707_prune"]; prune_config=next(c for c in grid["configurations"] if c["tag"]=="pc707_prune")
calibration=dict(primary)
calibration.update({
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "variant":"pc707_prune","base_strategy":"currentgf_nonempty",
 "selection_basis":"exact-707 case-held-out grouped-fold full-volume OOF; paired against currentgf and identical additive deployment",
 "Lesion_DSC":float(row["DSC"]),"Lesion_NSD":float(row["NSD"]),
 "metrics":str(Path(row["metrics"]).resolve()),
 "calibration_deployment":"anchor_currentgf",
 "source_deployment_modes":prune_config["source_deployment_modes"],
 "public_read":False,"hidden_read":False,"submission_created":False,
})
temporary=cal_path.with_name(f".{cal_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(calibration,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,cal_path)
qualified=base["decision"]=="PROMOTE" and additive["decision"]=="PROMOTE"
terminal={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),"status":"COMPLETE",
 "variant":"prune_composition","decision":"QUALIFY_FOR_FIXED_PUBLIC" if qualified else "KEEP_PRIMARY",
 "selection_basis":"exact-707 case-held-out grouped-fold full-volume OOF using cached primary predictions",
 "cases":707,"base_strategy":"currentgf_nonempty","summary":str(summary_path.resolve()),
 "calibration":str(cal_path.resolve()),"paired_vs_currentgf":str(base_path.resolve()),
 "paired_vs_additive":str(add_path.resolve()),"paired_vs_currentgf_payload":base,
 "paired_vs_additive_payload":additive,"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False,
}
temporary=terminal_path.with_name(f".{terminal_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(terminal,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,terminal_path)
print(json.dumps(terminal,indent=2))
PY
cat "${summary}"
