#!/usr/bin/env bash
set -euo pipefail

# Training-only audit of filled lung coverage cells. Proposals and the complete
# eight-source cache are frozen before complete training labels are opened.
# There is no GPU, public/hidden, packaging, or upload path in this controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
labels="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
base="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/base_plus_broad_e050"
base_report="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/oracle_e050/report.json"
output="${root}/analysis/lndb_lung_cell_headroom_20260823"
sparse="${output}/cells48_sparse"
complete_root="${output}/cells48_complete"
plan="${output}/PREDECLARED_PLAN.json"
decision="${output}/DECISION.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_lndb_lung_cell_headroom_20260823.lock"
flock -n 9 || { echo "LNDb lung-cell audit already active" >&2; exit 1; }
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
 "method":"filled 48mm occupied heuristic-lung cells in eight parity-separated sources",
 "rationale":"A filled cell directly overlaps lesions that point/blob seeds miss; eight-colour separation preserves one candidate component per adjacent cell",
 "parameters":{"target_spacing_mm":2.5,"cell_stride_mm":48.0,"sources":8,
               "bbox_context_mm":24.0,"tile_field_mm":96.0},
 "promotion_gate":{"minimum_LNDb_DSC_gain":0.20,"minimum_global_DSC_gain":0.028,
                    "maximum_mean_cells_per_LNDb_case":200.0,
                    "maximum_component_axis_extent_mm":48.000001},
 "manifest":str(manifest.resolve()),
 "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "frozen_base":str(base.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
fi

reuse_sparse=0
if "${python_bin}" - "${sparse}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); marker=root/"PROPOSALS_COMPLETE.json"
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text(encoding="utf-8"))
valid=(payload.get("cases")==30 and payload.get("labels_read") is False
 and float(payload.get("parameters",{}).get("target_spacing_mm",-1))==2.5
 and float(payload.get("parameters",{}).get("cell_stride_mm",-1))==48.0
 and float(payload.get("maximum_component_axis_extent_mm",1e9))<=48.000001)
for colour in range(8):
    valid &= len(list((root/f"colour{colour}").glob("*.nii.gz")))==30
raise SystemExit(0 if valid else 1)
PY
then
  reuse_sparse=1
  printf '%s reusing frozen exact-30 lung-cell proposal cache\n' "$(date -Ins)"
fi
if [[ "${reuse_sparse}" -eq 0 ]]; then
  "${python_bin}" -u "${root}/generate_lung_cell_proposals.py" \
    --manifest "${manifest}" --cohort chest:Chest_LNDb --output "${sparse}" \
    --target-spacing-mm 2.5 --cell-stride-mm 48 --expected-count 30 --workers 1 \
    --require-complete-labels > "${output}/generate.log" 2>&1
fi

proposal_args=(--proposal "${base}")
for colour in 0 1 2 3 4 5 6 7; do
  complete="${complete_root}/colour${colour}"
  "${python_bin}" -u "${root}/fuse_sparse_overlays.py" \
    --manifest "${manifest}" --base "${base}" --overlay "${sparse}/colour${colour}" \
    --empty-base --output "${complete}" \
    --marker "${output}/colour${colour}_complete.json" --workers 4 \
    > "${output}/colour${colour}_complete.log" 2>&1
  proposal_args+=(--proposal "${complete}")
done

# Every label-blind source is now frozen and exact-210. Labels are first opened here.
"${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
  --gt "${labels}" "${proposal_args[@]}" --component-policy source_separate \
  --output "${output}/oracle_cells48" --min-mm3 5 --max-mm3 1000000000000 \
  --tile-stride-mm 72 --margins-mm 24 --workers 4 --expected-count 210 \
  --manifest "${manifest}" > "${output}/oracle.log" 2>&1

"${python_bin}" - "${plan}" "${base_report}" "${output}/oracle_cells48/report.json" \
  "${sparse}/PROPOSALS_COMPLETE.json" "${complete_root}" "${decision}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
plan_path,base_path,report_path,generator_path,source,output=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text(encoding="utf-8"))
base=json.loads(base_path.read_text(encoding="utf-8"))
report=json.loads(report_path.read_text(encoding="utf-8"))
generator=json.loads(generator_path.read_text(encoding="utf-8"))
cohort="chest:Chest_LNDb"; margin="24"
baseline={"global_DSC":float(base["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(base["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_positive_tiles_per_case":float(base["margins"][margin]["mean_tiles_per_case"])}
candidate={"global_DSC":float(report["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(report["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_positive_tiles_per_case":float(report["margins"][margin]["mean_tiles_per_case"]),
 "mean_cells_per_LNDb_case":float(generator["cell_count"]["mean"]),
 "maximum_component_axis_extent_mm":float(generator["maximum_component_axis_extent_mm"]),
 "proposal_root":str(source.resolve()),"report":str(report_path.resolve())}
candidate["global_DSC_gain"]=candidate["global_DSC"]-baseline["global_DSC"]
candidate["LNDb_DSC_gain"]=candidate["LNDb_DSC"]-baseline["LNDb_DSC"]
gate=plan["promotion_gate"]
promote=(candidate["LNDb_DSC_gain"]>=gate["minimum_LNDb_DSC_gain"]
 and candidate["global_DSC_gain"]>=gate["minimum_global_DSC_gain"]
 and candidate["mean_cells_per_LNDb_case"]<=gate["maximum_mean_cells_per_LNDb_case"]
 and candidate["maximum_component_axis_extent_mm"]<=gate["maximum_component_axis_extent_mm"])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "warning":"Complete-training-label proposal oracle only; never deploy oracle masks.",
 "plan":str(plan_path.resolve()),"baseline":baseline,"candidate":candidate,
 "decision":"PROMOTE_LNDB_LUNG_CELLS" if promote else "KEEP_BROAD_PROMPT_BANK",
 "labels_read_after_proposals_frozen":True,"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
