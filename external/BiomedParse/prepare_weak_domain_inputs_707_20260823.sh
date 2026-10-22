#!/usr/bin/env bash
set -euo pipefail

# CPU-only, label-blind preparation of the two low-recall cohorts in the
# balanced 707-case expansion. GPU proposal inference is a separate gated
# stage. The broad prompt banks were declared by the 210-case recall-extension
# experiment before its labels were evaluated.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
manifest="${project}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/manifest.json"
output="${root}/analysis/weak_domain_biomedparse_707_20260823"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "707 weak-domain input preparation already active" >&2; exit 1; }
[[ -s "${output}/INPUTS_COMPLETE.json" ]] && exit 0
[[ -s "${manifest}" ]] || { echo "missing manifest: ${manifest}" >&2; exit 2; }

"${python_bin}" - "${manifest}" "${output}" <<'PY'
import hashlib,json,sys
from pathlib import Path

manifest,output=map(Path,sys.argv[1:])
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
cohorts={"autopet":"abdomen:WholeBody_autoPETCT","lndb":"chest:Chest_LNDb"}
for tag,cohort in cohorts.items():
    selected=sorted(
        (item for item in items if item["cohort"]==cohort),
        key=lambda item:str(item["name"]),
    )
    if len(selected)!=101 or any(item.get("complete_label") is not True for item in selected):
        raise SystemExit(f"unexpected complete-label cohort coverage: {tag} {len(selected)}")
    images=output/tag/"images"
    images.mkdir(parents=True,exist_ok=True)
    expected=set()
    for item in selected:
        source=Path(item["image"]).resolve(strict=True)
        target=images/source.name
        expected.add(source.name)
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source:
                raise SystemExit(f"collision: {target}")
        else:
            target.symlink_to(source)
    observed={path.name for path in images.glob("*.nii.gz")}
    if observed!=expected:
        raise SystemExit(f"image coverage mismatch: {tag} {len(observed)}/{len(expected)}")
    (output/tag/"selection.json").write_text(json.dumps({
        "cohort":cohort,
        "cases":101,
        "labels_read":False,
        "manifest":str(manifest.resolve()),
        "items":[{
            "name":item["name"],
            "image":item["image"],
            "prediction_name":item["prediction_name"],
        } for item in selected],
    },indent=2)+"\n",encoding="utf-8")
PY

prepare_bank() {
  local tag="$1" cohort="$2"; shift 2
  local prompt_args=() prompt
  for prompt in "$@"; do prompt_args+=(--prompt "${prompt}"); done
  mkdir -p "${output}/${tag}/inputs"
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py prepare \
    --images "${output}/${tag}/images" --output "${output}/${tag}/inputs" \
    "${prompt_args[@]}" --a-min -160 --a-max 240 --continue-prediction \
    > "${output}/${tag}/prepare.log" 2>&1
  "${python_bin}" -u "${root}/audit_miccai_inputs.py" \
    --manifest "${manifest}" --cohort "${cohort}" \
    --images "${output}/${tag}/images" --inputs "${output}/${tag}/inputs" \
    "${prompt_args[@]}" --expected-cases 101 --workers 2 \
    --output "${output}/${tag}/INPUT_AUDIT.json" \
    > "${output}/${tag}/audit.log" 2>&1
}

prepare_bank autopet abdomen:WholeBody_autoPETCT \
  "whole body lesion" "tumor" "malignant tumor" "PET-avid lesion" \
  "lymph node metastasis" "bone metastasis" "liver metastasis" \
  "lung metastasis"
prepare_bank lndb chest:Chest_LNDb \
  "pulmonary nodule" "lung nodule" "lung mass" "lung tumor" "lung cancer"

"${python_bin}" - "${manifest}" "${output}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path

manifest,output=map(Path,sys.argv[1:])
audits={
    tag:json.loads((output/tag/"INPUT_AUDIT.json").read_text(encoding="utf-8"))
    for tag in ("autopet","lndb")
}
if any(report.get("cases")!=101 or report.get("labels_read") is not False for report in audits.values()):
    raise SystemExit("707 weak-domain input audit did not pass")
digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
payload={
    "schema_version":1,
    "created_at":datetime.now().astimezone().isoformat(),
    "manifest":str(manifest.resolve()),
    "manifest_sha256":digest,
    "cases_per_cohort":101,
    "total_cases":202,
    "cohorts":audits,
    "labels_read":False,
    "gpu_used":False,
    "purpose":"prewarm inputs only; prompt inference and threshold selection remain gated",
}
temporary=output/".INPUTS_COMPLETE.json.tmp"
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output/"INPUTS_COMPLETE.json")
print(json.dumps(payload,indent=2))
PY
