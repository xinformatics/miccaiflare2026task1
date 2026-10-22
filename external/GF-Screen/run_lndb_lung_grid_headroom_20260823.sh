#!/usr/bin/env bash
set -euo pipefail

# Single-configuration, training-only pulmonary coverage-grid audit. The
# geometry is derived from the fixed bbox+24 deployment context and frozen
# before labels are scored. No GPU, public/hidden data, package, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
labels="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
base="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/base_plus_broad_e050"
base_report="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/oracle_e050/report.json"
output="${root}/analysis/lndb_lung_grid_headroom_20260823"
sparse="${output}/grid48_sparse"
complete_source="${output}/grid48_complete"
plan="${output}/PREDECLARED_PLAN.json"
decision="${output}/DECISION.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_lndb_lung_grid_headroom_20260823.lock"
flock -n 9 || { echo "LNDb lung-grid audit already active" >&2; exit 1; }
[[ -s "${decision}" ]] && exit 0
for required in "${manifest}" "${base_report}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${labels}" "${base}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done

if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${manifest}" "${base}" "${plan}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
manifest,base,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "labels_read":False,"cohort":"chest:Chest_LNDb","cases":30,
 "method":"one seed at the center of every occupied 48mm heuristic-lung cell",
 "rationale":"48mm cell width equals twice the frozen 24mm bbox context, providing axis-wise lung coverage independent of lesion appearance",
 "parameters":{"target_spacing_mm":2.5,"grid_spacing_mm":48.0,"sphere_radius_mm":3.0},
 "component_policy":"source_separate_exact_dedup",
 "promotion_gate":{"minimum_LNDb_DSC_gain":0.10,"minimum_global_DSC_gain":0.014,
                    "maximum_mean_grid_seeds_per_LNDb_case":100.0,
                    "maximum_global_mean_tiles_per_case":40.0},
 "manifest":str(manifest.resolve()),
 "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "frozen_base":str(base.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
fi

"${python_bin}" -u "${root}/generate_lung_grid_proposals.py" \
  --manifest "${manifest}" --cohort chest:Chest_LNDb --output "${sparse}" \
  --target-spacing-mm 2.5 --grid-spacing-mm 48 --sphere-radius-mm 3 \
  --expected-count 30 --workers 2 > "${output}/generate.log" 2>&1

"${python_bin}" -u "${root}/fuse_sparse_overlays.py" \
  --manifest "${manifest}" --base "${base}" --overlay "${sparse}" \
  --empty-base --output "${complete_source}" \
  --marker "${output}/grid48_complete.json" --workers 4 \
  > "${output}/complete_source.log" 2>&1

# Both proposal sources are now frozen. Labels are opened only for the oracle.
"${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
  --gt "${labels}" --proposal "${base}" --proposal "${complete_source}" \
  --component-policy source_separate --output "${output}/oracle_grid48" \
  --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
  --margins-mm 24 --workers 4 --expected-count 210 --manifest "${manifest}" \
  > "${output}/oracle.log" 2>&1

"${python_bin}" - "${plan}" "${base_report}" "${output}/oracle_grid48/report.json" \
  "${sparse}/PROPOSALS_COMPLETE.json" "${complete_source}" "${decision}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
plan_path,base_path,report_path,generator_path,source,output=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text(encoding="utf-8")); base=json.loads(base_path.read_text(encoding="utf-8"))
report=json.loads(report_path.read_text(encoding="utf-8")); generator=json.loads(generator_path.read_text(encoding="utf-8"))
cohort="chest:Chest_LNDb"; margin="24"
baseline={"global_DSC":float(base["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(base["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_tiles_per_case":float(base["margins"][margin]["mean_tiles_per_case"])}
candidate={"global_DSC":float(report["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(report["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_tiles_per_case":float(report["margins"][margin]["mean_tiles_per_case"]),
 "mean_grid_seeds_per_LNDb_case":float(generator["grid_seed_count"]["mean"]),
 "proposal_source":str(source.resolve()),"report":str(report_path.resolve())}
candidate["global_DSC_gain"]=candidate["global_DSC"]-baseline["global_DSC"]
candidate["LNDb_DSC_gain"]=candidate["LNDb_DSC"]-baseline["LNDb_DSC"]
gate=plan["promotion_gate"]
promote=(candidate["LNDb_DSC_gain"]>=gate["minimum_LNDb_DSC_gain"]
 and candidate["global_DSC_gain"]>=gate["minimum_global_DSC_gain"]
 and candidate["mean_grid_seeds_per_LNDb_case"]<=gate["maximum_mean_grid_seeds_per_LNDb_case"]
 and candidate["mean_tiles_per_case"]<=gate["maximum_global_mean_tiles_per_case"])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "warning":"Complete-training-label proposal oracle only; never deploy oracle masks.",
 "plan":str(plan_path.resolve()),"baseline":baseline,"candidate":candidate,
 "decision":"PROMOTE_LNDB_GRID_SOURCE" if promote else "KEEP_BROAD_PROMPT_BANK",
 "labels_read_after_proposals_frozen":True,"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
