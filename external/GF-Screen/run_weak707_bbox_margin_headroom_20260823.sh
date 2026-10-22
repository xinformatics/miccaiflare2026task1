#!/usr/bin/env bash
set -euo pipefail

# Training-only context-margin selection on the frozen exact-210 proposal bank.
# Proposals were generated without labels. Complete training labels are read
# only by the oracle below; public/hidden data, packaging, and upload are absent.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
proposal="${project}/external/BiomedParse/analysis/weak_domain_biomedparse_recall_extension_20260823/enriched_plus_initial_e050"
labels="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
output="${root}/analysis/weak707_bbox_margin_headroom_20260823"
oracle="${output}/oracle"
decision="${output}/DECISION.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_weak707_bbox_margin_headroom_20260823.lock"
flock -n 9 || { echo "bbox margin headroom controller already active" >&2; exit 1; }
[[ -s "${decision}" ]] && exit 0
for required in "${proposal}" "${labels}" "${manifest}"; do
  [[ -e "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
  --gt "${labels}" --proposal "${proposal}" --output "${oracle}" \
  --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
  --margins-mm 16 24 32 48 72 96 --workers 4 --expected-count 210 \
  --manifest "${manifest}" > "${output}/oracle.log" 2>&1

"${python_bin}" - "${oracle}/report.json" "${decision}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
report_path,output=map(Path,sys.argv[1:])
report=json.loads(report_path.read_text(encoding="utf-8"))
auto="abdomen:WholeBody_autoPETCT"; lndb="chest:Chest_LNDb"
rows=[]
for margin_text,global_row in sorted(report["margins"].items(),key=lambda item:int(item[0])):
    margin=int(margin_text)
    cohorts=report["cohort_summary"]
    a=float(cohorts[auto]["margins"][margin_text]["macro_dsc"])
    l=float(cohorts[lndb]["margins"][margin_text]["macro_dsc"])
    rows.append({
      "margin_mm":margin,"macro_DSC":float(global_row["macro_dsc"]),
      "autoPETCT_DSC":a,"LNDb_DSC":l,"weak_domain_floor_DSC":min(a,l),
      "mean_tiles_per_case":float(global_row["mean_tiles_per_case"]),
      "total_tiles":int(global_row["total_tiles"]),
    })
baseline=next(row for row in rows if row["margin_mm"]==24)
best_floor=max(row["weak_domain_floor_DSC"] for row in rows)
near=[row for row in rows if row["weak_domain_floor_DSC"]>=best_floor-0.005]
selected=min(near,key=lambda row:(row["mean_tiles_per_case"],row["margin_mm"]))
selected["macro_DSC_gain_over_24"]=selected["macro_DSC"]-baseline["macro_DSC"]
selected["autoPETCT_DSC_gain_over_24"]=selected["autoPETCT_DSC"]-baseline["autoPETCT_DSC"]
selected["LNDb_DSC_gain_over_24"]=selected["LNDb_DSC"]-baseline["LNDb_DSC"]
use_expanded=(
  selected["margin_mm"]>24 and
  (selected["LNDb_DSC_gain_over_24"]>=0.03 or selected["macro_DSC_gain_over_24"]>=0.01)
)
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "warning":"Complete-training-label proposal oracle only; never deploy oracle masks.",
 "selection_rule":"fewest mean tiles within 0.005 of best autoPET/LNDb floor DSC",
 "baseline_margin_mm":24,"candidates":rows,"selected":selected,
 "decision":"EXPAND_CONTEXT" if use_expanded else "KEEP_MARGIN_24",
 "effective_margin_mm":selected["margin_mm"] if use_expanded else 24,
 "proposal":report["proposals"],"oracle_report":str(report_path.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
