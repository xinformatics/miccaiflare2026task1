#!/usr/bin/env bash
set -euo pipefail

# CPU-only preparation for two predeclared low-recall complete-label cohorts.
# Labels are not opened. GPU inference is a separate serialized stage.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
manifest="${project}/external/GF-Screen/analysis/fpr_training_mining_20260822/manifest.json"
output="${root}/analysis/weak_domain_biomedparse_20260823"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "weak-domain BiomedParse input preparation already active" >&2; exit 1; }
[[ -s "${output}/INPUTS_COMPLETE.txt" ]] && exit 0

"${python_bin}" - "${manifest}" "${output}" <<'PY'
import json,sys
from pathlib import Path
manifest,output=Path(sys.argv[1]),Path(sys.argv[2])
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
cohorts={"autopet":"abdomen:WholeBody_autoPETCT","lndb":"chest:Chest_LNDb"}
for tag,cohort in cohorts.items():
    selected=sorted((item for item in items if item["cohort"]==cohort),key=lambda item:item["name"])
    if len(selected)!=30 or any(item.get("complete_label") is not True for item in selected):
        raise SystemExit(f"unexpected cohort coverage: {tag} {len(selected)}")
    images=output/tag/"images"; images.mkdir(parents=True,exist_ok=True)
    for item in selected:
        source=Path(item["image"]).resolve(strict=True); target=images/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source: raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source)
    (output/tag/"selection.json").write_text(json.dumps({
        "cohort":cohort,"cases":30,"labels_read":False,
        "items":[{"name":item["name"],"image":item["image"],"prediction_name":item["prediction_name"]} for item in selected],
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
    "${prompt_args[@]}" --expected-cases 30 --workers 2 \
    --output "${output}/${tag}/INPUT_AUDIT.json" \
    > "${output}/${tag}/audit.log" 2>&1
}

prepare_bank autopet abdomen:WholeBody_autoPETCT \
  "malignant lesion" "metastasis" "cancer" "abnormal mass"
prepare_bank lndb chest:Chest_LNDb "lung tumor" "lung nodule"
date -Ins > "${output}/INPUTS_COMPLETE.txt"
