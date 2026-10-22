#!/usr/bin/env bash
set -euo pipefail

# CPU-only, label-blind cache preparation for the source-separated transfer
# diagnostic.  It may create label symlinks through the shared OOF staging
# utility, but neither preparation nor cache audit opens those labels.  GPU
# inference and metric evaluation are intentionally absent from this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v2_training_20260822"
index="${train}/patches_bbox24_min5_v5/index.csv"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
epoch105="${train}/epoch105_abd_g090_l050"
currentgf="${root}/analysis/candidate_refiner_v1_training_20260822/gf_raw_l050"
primary_staging="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823/staging"
union_staging="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging"
audit_marker="${root}/analysis/candidate_refiner_v3_source_separate_20260823/COMPONENT_POLICY_AUDIT_COMPLETE.json"
oracle="${root}/analysis/candidate_bbox_context_oracle_source_separate_train210_20260823/report.json"
output="${root}/analysis/candidate_refiner_v3_source_separate_oof_20260823"
current_staging="${output}/staging_currentgf"
complete="${output}/PREWARM_COMPLETE.json"

mkdir -p "${output}"
exec 9>"${root}/runs/prewarm_candidate_refiner_source_separate_oof_20260823.lock"
flock -n 9 || { echo "source-separated OOF prewarm already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
for required in "${index}" "${manifest}" "${audit_marker}" "${oracle}" \
  "${primary_staging}/STAGING_COMPLETE.json" "${union_staging}/STAGING_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${audit_marker}" "${oracle}" <<'PY'
import json,sys
audit=json.load(open(sys.argv[1],encoding="utf-8"))
oracle=json.load(open(sys.argv[2],encoding="utf-8"))
if audit.get("cases") != 210 or audit.get("labels_read") is not False:
    raise SystemExit("source-separated connectivity audit is incomplete")
if oracle.get("cases") != 210 or oracle.get("component_policy") != "source_separate":
    raise SystemExit("source-separated bbox oracle is incomplete")
if float(oracle["margins"]["24"]["macro_dsc"]) < 0.75:
    raise SystemExit("source-separated proposal ceiling is too low for transfer")
PY

if [[ ! -s "${current_staging}/STAGING_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
    --index "${index}" --manifest "${manifest}" --proposals "${currentgf}" \
    --output "${current_staging}" --folds 5 --seed 20260822 \
    > "${output}/stage_currentgf.log" 2>&1
fi

"${python_bin}" - \
  "${primary_staging}/STAGING_COMPLETE.json" \
  "${union_staging}/STAGING_COMPLETE.json" \
  "${current_staging}/STAGING_COMPLETE.json" <<'PY'
import json,sys
reports=[json.load(open(path,encoding="utf-8")) for path in sys.argv[1:]]
for report in reports:
    if report.get("cases") != 210 or report.get("exact_case_coverage_once") is not True:
        raise SystemExit("OOF staging is incomplete")
reference=[(row["fold"],row["cases"],row["zero_patch_cases"]) for row in reports[0]["folds"]]
for report in reports[1:]:
    observed=[(row["fold"],row["cases"],row["zero_patch_cases"]) for row in report["folds"]]
    if observed != reference:
        raise SystemExit("source OOF partitions differ")
PY

fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }
prepare_fold() {
  local fold="$1" work
  work="$(fold_work "${fold}")"
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${primary_staging}/fold${fold}/images" \
      --proposal "epoch105=${primary_staging}/fold${fold}/proposals" \
      --proposal "currentgf=${current_staging}/fold${fold}/proposals" \
      --component-policy source_separate \
      --base "${union_staging}/fold${fold}/proposals" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 \
      --bbox-margin-mm 24 --tile-stride-mm 72 \
      > "${output}/fold${fold}.prepare.log" 2>&1
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 1 \
    > "${output}/fold${fold}.audit.log" 2>&1
}
prepare_sequence() { local fold; for fold in "$@"; do prepare_fold "${fold}"; done; }
prepare_sequence 0 2 4 & prep0=$!
prepare_sequence 1 3 & prep1=$!
status=0; wait "${prep0}" || status=$?; wait "${prep1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

"${python_bin}" - "${output}" "${complete}" <<'PY'
import datetime,json,sys
from pathlib import Path
root,output=map(Path,sys.argv[1:])
folds=[]; cases=[]; candidates=0
for fold in range(5):
    work=root/f"fold{fold}"/"work_bbox24"
    manifest=json.loads((work/"manifest.json").read_text(encoding="utf-8"))
    audit=json.loads((work/"AUDIT.json").read_text(encoding="utf-8"))
    if manifest.get("component_policy") != "source_separate" or audit.get("component_policy") != "source_separate":
        raise SystemExit(f"fold {fold} did not preserve source-separated policy")
    fold_cases=[row["case"] for row in manifest["case_records"]]
    if len(fold_cases)!=len(set(fold_cases)) or audit.get("exact_file_coverage") is not True:
        raise SystemExit(f"fold {fold} cache audit failed")
    cases.extend(fold_cases); candidates+=int(manifest["candidates"])
    folds.append({"fold":fold,"cases":len(fold_cases),"candidates":int(manifest["candidates"]),"audit":str((work/"AUDIT.json").resolve())})
if len(cases)!=210 or len(set(cases))!=210:
    raise SystemExit("source-separated OOF caches do not cover exact 210 cases")
report={
    "schema_version":1,
    "created_at":datetime.datetime.now().astimezone().isoformat(),
    "labels_read":False,
    "gpu_used":False,
    "component_policy":"source_separate",
    "cases":210,
    "candidates":candidates,
    "folds":folds,
}
output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
PY
