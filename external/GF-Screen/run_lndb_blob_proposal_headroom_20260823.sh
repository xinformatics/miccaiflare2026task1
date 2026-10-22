#!/usr/bin/env bash
set -euo pipefail

# Bounded training-only LNDb proposal recall pilot. The classical CT proposal
# parameters and selection rule are frozen before any complete label is scored.
# No GPU, public/hidden data, packaging, or upload path is present.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
labels="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
base="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/base_plus_broad_e050"
base_report="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/oracle_e050/report.json"
output="${root}/analysis/lndb_blob_proposal_headroom_20260823"
proposals="${output}/proposals"
plan="${output}/PREDECLARED_PLAN.json"
decision="${output}/DECISION.json"
top_values=(50 100 200)

mkdir -p "${output}"
exec 9>"${root}/runs/run_lndb_blob_proposal_headroom_20260823.lock"
flock -n 9 || { echo "LNDb blob proposal audit already active" >&2; exit 1; }
[[ -s "${decision}" ]] && exit 0
for required in "${manifest}" "${base_report}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${labels}" "${base}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]

if [[ ! -s "${plan}" ]]; then
  "${python_bin}" - "${manifest}" "${base}" "${plan}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
manifest,base,output=map(Path,sys.argv[1:])
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "labels_read":False,"cohort":"chest:Chest_LNDb","cases":30,
 "method":"multiscale scale-normalized 3D LoG inside heuristic lung mask",
 "parameters":{"top_k":[50,100,200],"target_spacing_mm":2.5,
  "sigmas_mm":[2.5,5.0,7.5],"response_threshold":0.025,
  "nms_distance_mm":6.0,"sphere_radius_mm":4.0},
 "selection_rule":"fewest candidates within 0.005 LNDb bbox+24 DSC of the best",
 "promotion_gate":{"minimum_LNDb_DSC_gain":0.05,"minimum_global_DSC_gain":0.007},
 "manifest":str(manifest.resolve()),
 "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "frozen_base":str(base.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
fi

"${python_bin}" -u "${root}/generate_lung_blob_proposals.py" \
  --manifest "${manifest}" --cohort chest:Chest_LNDb --output "${proposals}" \
  --top-k 50,100,200 --target-spacing 2.5 --sigmas-mm 2.5,5.0,7.5 \
  --response-threshold 0.025 --nms-distance-mm 6 --sphere-radius-mm 4 \
  --expected-count 30 --workers 2 > "${output}/generate.log" 2>&1

# Proposal generation is frozen. Complete labels are opened only below to
# measure the attainable bbox+24 ceiling of each fixed candidate budget.
for top in "${top_values[@]}"; do
  tag=$(printf '%03d' "${top}")
  fused="${output}/base_plus_blob_top${tag}"
  "${python_bin}" -u "${root}/fuse_sparse_overlays.py" \
    --manifest "${manifest}" --base "${base}" \
    --overlay "${proposals}/top${tag}" --output "${fused}" \
    --marker "${output}/base_plus_blob_top${tag}.json" --workers 4 \
    > "${output}/fuse_top${tag}.log" 2>&1
  "${python_bin}" -u "${root}/analyze_candidate_bbox_context_oracle.py" \
    --gt "${labels}" --proposal "${fused}" --output "${output}/oracle_top${tag}" \
    --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
    --margins-mm 24 --workers 4 --expected-count 210 --manifest "${manifest}" \
    > "${output}/oracle_top${tag}.log" 2>&1
done

"${python_bin}" - "${plan}" "${base_report}" "${output}" "${decision}" <<'PY'
import json,os,sys
from datetime import datetime
from pathlib import Path
plan_path,base_path,root,output=map(Path,sys.argv[1:])
plan=json.loads(plan_path.read_text(encoding="utf-8"))
base=json.loads(base_path.read_text(encoding="utf-8"))
cohort="chest:Chest_LNDb"
baseline={"global_DSC":float(base["margins"]["24"]["macro_dsc"]),
          "LNDb_DSC":float(base["cohort_summary"][cohort]["margins"]["24"]["macro_dsc"]),
          "mean_tiles_per_case":float(base["margins"]["24"]["mean_tiles_per_case"])}
rows=[]
for count in plan["parameters"]["top_k"]:
    tag=f"{int(count):03d}"
    report_path=root/f"oracle_top{tag}"/"report.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    rows.append({"top_k":int(count),
      "global_DSC":float(report["margins"]["24"]["macro_dsc"]),
      "LNDb_DSC":float(report["cohort_summary"][cohort]["margins"]["24"]["macro_dsc"]),
      "mean_tiles_per_case":float(report["margins"]["24"]["mean_tiles_per_case"]),
      "proposal":str((root/f"base_plus_blob_top{tag}").resolve()),
      "report":str(report_path.resolve())})
best=max(row["LNDb_DSC"] for row in rows)
near=[row for row in rows if row["LNDb_DSC"]>=best-0.005]
selected=min(near,key=lambda row:(row["top_k"],row["mean_tiles_per_case"]))
selected["global_DSC_gain"]=selected["global_DSC"]-baseline["global_DSC"]
selected["LNDb_DSC_gain"]=selected["LNDb_DSC"]-baseline["LNDb_DSC"]
promote=(selected["global_DSC_gain"]>=plan["promotion_gate"]["minimum_global_DSC_gain"]
         and selected["LNDb_DSC_gain"]>=plan["promotion_gate"]["minimum_LNDb_DSC_gain"])
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "warning":"Complete-training-label proposal oracle only; never deploy oracle masks.",
 "plan":str(plan_path.resolve()),"baseline":baseline,"candidates":rows,
 "selected":selected,"decision":"PROMOTE_LNDB_BLOB_SOURCE" if promote else "KEEP_BROAD_PROMPT_BANK",
 "labels_read_after_proposals_frozen":True,"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY
