#!/usr/bin/env bash
set -euo pipefail

# Scale the frozen, label-blind 48 mm lung-cell proposal rule to all 707
# complete-label development cases, then measure its proposal-only ceiling.
# The proposal cache is completely frozen before labels are linked or opened.
# This controller has no public/hidden inference, packaging, or upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
pilot="${root}/analysis/lndb_lung_cell_headroom_20260823/DECISION.json"
epoch105="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050"
currentgf="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
weak_marker="${project}/external/BiomedParse/analysis/weak_domain_biomedparse_707_selected_20260823/PROPOSALS_COMPLETE.json"
output="${root}/analysis/lung_cell707_scale_20260823"
cells="${output}/cells48_all707"
labels="${output}/labels707"
plan="${output}/PREDECLARED_PLAN.json"
audit="${output}/proposal_sources_audit.json"
baseline_report="${output}/oracle_baseline/report.json"
candidate_report="${output}/oracle_cells48/report.json"
decision="${output}/DECISION.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_lung_cell707_scale_and_oracle_20260823.lock"
flock -n 9 || { echo "lung-cell 707 scale controller already active" >&2; exit 1; }
[[ -s "${decision}" ]] && exit 0
for required in "${manifest}" "${pilot}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

pilot_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${pilot}")
[[ "${pilot_decision}" == PROMOTE_LNDB_LUNG_CELLS ]] || {
  echo "pilot did not promote lung cells: ${pilot_decision}" >&2
  exit 2
}

# Freeze the scale-out gate before any complete labels are linked or opened.
if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${manifest}" "${pilot}" "${plan}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
manifest,pilot,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "labels_read":False,"cases":707,
 "method":"filled 48mm occupied heuristic-lung cells in eight parity-separated sources",
 "parameters":{"target_spacing_mm":2.5,"cell_stride_mm":48.0,"sources":8,
               "bbox_context_mm":24.0,"tile_field_mm":96.0},
 "promotion_gate":{"minimum_LNDb_DSC_gain":0.20,
                    "minimum_global_DSC_gain":0.02,
                    "maximum_mean_cells_per_case":200.0,
                    "maximum_component_axis_extent_mm":48.000001},
 "pilot_decision":str(pilot.resolve()),
 "manifest":str(manifest.resolve()),
 "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
fi

# This generation is label blind. Four workers were deliberately avoided after
# earlier host-memory pressure; eight workers use the measured host headroom
# while atomic per-case sidecars make a service restart resumable.
if [[ ! -s "${cells}/PROPOSALS_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/generate_lung_cell_proposals.py" \
    --manifest "${manifest}" --all-cases --output "${cells}" \
    --target-spacing-mm 2.5 --cell-stride-mm 48 --expected-count 707 \
    --workers 8 --require-complete-labels --resume > "${output}/generate.log" 2>&1
fi

# Require exact geometry/content coverage without opening any label volume.
source_args=()
for colour in 0 1 2 3 4 5 6 7; do
  source_args+=(--source "lung_cell${colour}=${cells}/colour${colour}")
done
if [[ ! -s "${audit}" ]]; then
  "${python_bin}" -u "${root}/audit_training_proposal_sources.py" \
    --manifest "${manifest}" "${source_args[@]}" --expected-count 707 \
    --workers 4 --output "${audit}" > "${output}/audit.log" 2>&1
fi

# Only now link the complete labels named by the manifest. The oracle scripts
# read these links; the proposal generator and source audit above never do.
"${python_bin}" - "${manifest}" "${labels}" <<'PY'
import json,os,sys
from pathlib import Path
manifest,output=map(Path,sys.argv[1:]); output.mkdir(parents=True,exist_ok=True)
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
expected=set()
for item in items:
    if item.get("complete_label") is not True:
        raise SystemExit("incomplete label in exact-707 manifest")
    name=str(item["prediction_name"]); source=Path(item["label"]).resolve(strict=True)
    target=output/name; expected.add(name)
    if target.is_symlink() or target.exists():
        if target.resolve(strict=True)!=source: raise SystemExit(f"label collision: {target}")
    else: target.symlink_to(source)
observed={p.name for p in output.glob("*.nii.gz")}
if len(expected)!=707 or observed!=expected:
    raise SystemExit(f"label coverage mismatch {len(observed)}/{len(expected)}")
PY

# Wait for the exact four-source base cache. Cell generation itself does not
# need those sources, so the expensive CPU work can overlap GF inference.
while [[ ! -s "${epoch105}/PROPOSALS_COMPLETE.json" \
    || ! -s "${currentgf}/PROPOSALS_COMPLETE.json" || ! -s "${weak_marker}" ]]; do
  printf '%s lung cells frozen; waiting for exact four-source base cache\n' "$(date -Ins)"
  sleep 30
done
autopet=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["banks"]["autopet"]["complete_source"])' \
  "${weak_marker}")
lndb=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["banks"]["lndb"]["complete_source"])' \
  "${weak_marker}")
base_args=(--proposal "${epoch105}" --proposal "${currentgf}" \
  --proposal "${autopet}" --proposal "${lndb}")
cell_args=()
for colour in 0 1 2 3 4 5 6 7; do
  cell_args+=(--proposal "${cells}/colour${colour}")
done

if [[ ! -s "${baseline_report}" ]]; then
  "${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
    --gt "${labels}" "${base_args[@]}" --component-policy source_separate \
    --output "${output}/oracle_baseline" --min-mm3 5 \
    --max-mm3 1000000000000 --tile-stride-mm 72 --margins-mm 24 \
    --workers 8 --expected-count 707 --manifest "${manifest}" \
    > "${output}/oracle_baseline.log" 2>&1
fi
if [[ ! -s "${candidate_report}" ]]; then
  "${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
    --gt "${labels}" "${base_args[@]}" "${cell_args[@]}" \
    --component-policy source_separate --output "${output}/oracle_cells48" \
    --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
    --margins-mm 24 --workers 16 --expected-count 707 --manifest "${manifest}" \
    > "${output}/oracle_cells48.log" 2>&1
fi

"${python_bin}" - "${plan}" "${cells}/PROPOSALS_COMPLETE.json" \
  "${baseline_report}" "${candidate_report}" "${audit}" "${decision}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
plan_path,cells_path,base_path,candidate_path,audit_path,output=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text()); cells=json.loads(cells_path.read_text())
base=json.loads(base_path.read_text()); candidate=json.loads(candidate_path.read_text())
cohort="chest:Chest_LNDb"; margin="24"
baseline={"global_DSC":float(base["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(base["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_positive_tiles_per_case":float(base["margins"][margin]["mean_tiles_per_case"])}
result={"global_DSC":float(candidate["margins"][margin]["macro_dsc"]),
 "LNDb_DSC":float(candidate["cohort_summary"][cohort]["margins"][margin]["macro_dsc"]),
 "mean_positive_tiles_per_case":float(candidate["margins"][margin]["mean_tiles_per_case"]),
 "mean_cells_per_case":float(cells["cell_count"]["mean"]),
 "maximum_component_axis_extent_mm":float(cells["maximum_component_axis_extent_mm"]),
 "proposal_root":str(cells_path.parent.resolve()),"report":str(candidate_path.resolve())}
result["global_DSC_gain"]=result["global_DSC"]-baseline["global_DSC"]
result["LNDb_DSC_gain"]=result["LNDb_DSC"]-baseline["LNDb_DSC"]
gate=plan["promotion_gate"]
promote=(result["LNDb_DSC_gain"]>=gate["minimum_LNDb_DSC_gain"]
 and result["global_DSC_gain"]>=gate["minimum_global_DSC_gain"]
 and result["mean_cells_per_case"]<=gate["maximum_mean_cells_per_case"]
 and result["maximum_component_axis_extent_mm"]<=gate["maximum_component_axis_extent_mm"])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "warning":"Complete-training-label proposal oracle only; never deploy oracle masks.",
 "plan":str(plan_path.resolve()),"baseline":baseline,"candidate":result,
 "decision":"PROMOTE_LUNG_CELLS_707" if promote else "KEEP_FOUR_SOURCE_707",
 "proposal_source_audit":str(audit_path.resolve()),
 "labels_read_after_proposals_frozen":True,"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
