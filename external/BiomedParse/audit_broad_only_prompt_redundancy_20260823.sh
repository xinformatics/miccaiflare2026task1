#!/usr/bin/env bash
set -euo pipefail

# Training-label-only ablation: determine whether the initial NMS-enabled
# prompt bank adds meaningful proposal coverage beyond the selected broad,
# no-NMS bank. This never reads public/hidden data and never packages output.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${gf}/analysis/fpr_training_mining_20260822/manifest.json"
base="${gf}/analysis/candidate_refiner_v2_multisource_20260823/proposal_union"
labels="${gf}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
broad="${root}/analysis/weak_domain_biomedparse_recall_extension_20260823"
full_report="${broad}/oracle_e050/report.json"
output="${root}/analysis/weak_domain_broad_only_ablation_20260823"
proposal="${output}/base_plus_broad_e050"
oracle="${output}/oracle_e050"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "broad-only prompt ablation already active" >&2; exit 1; }
[[ -s "${output}/ABLATION_COMPLETE.txt" ]] && exit 0

for required in "${manifest}" "${full_report}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]
[[ $(find "${broad}/autopet_e050_nifti" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 30 ]]
[[ $(find "${broad}/lndb_e050_nifti" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 30 ]]

"${python_bin}" -u "${gf}/fuse_sparse_overlays.py" \
  --manifest "${manifest}" --base "${base}" \
  --overlay "${broad}/autopet_e050_nifti" \
  --overlay "${broad}/lndb_e050_nifti" \
  --output "${proposal}" --marker "${output}/base_plus_broad_e050.json" \
  --workers 4 > "${output}/fusion.log" 2>&1

"${python_bin}" -u "${gf}/analyze_candidate_bbox_context_oracle.py" \
  --gt "${labels}" --proposal "${proposal}" --output "${oracle}" \
  --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
  --margins-mm 24 --workers 4 --expected-count 210 --manifest "${manifest}" \
  > "${output}/oracle.log" 2>&1

"${python_bin}" - "${oracle}/report.json" "${full_report}" "${output}/DECISION.json" <<'PY'
import datetime
import json
import sys
from pathlib import Path

broad_path, full_path, destination = map(Path, sys.argv[1:])
broad = json.loads(broad_path.read_text(encoding="utf-8"))
full = json.loads(full_path.read_text(encoding="utf-8"))
names = ("abdomen:WholeBody_autoPETCT", "chest:Chest_LNDb")

def values(report):
    return {
        "macro_DSC": float(report["margins"]["24"]["macro_dsc"]),
        "mean_tiles_per_case": float(report["margins"]["24"]["mean_tiles_per_case"]),
        "autoPETCT_DSC": float(report["cohort_summary"][names[0]]["margins"]["24"]["macro_dsc"]),
        "LNDb_DSC": float(report["cohort_summary"][names[1]]["margins"]["24"]["macro_dsc"]),
    }

broad_values = values(broad)
full_values = values(full)
deltas = {key: full_values[key] - broad_values[key] for key in broad_values}
# Retain the older bank only if it contributes at least 0.005 Dice globally or
# to either weak cohort. Tile count is reported but does not drive this recall
# decision.
retain = any(deltas[key] >= 0.005 for key in ("macro_DSC", "autoPETCT_DSC", "LNDb_DSC"))
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "warning": "Complete-training-label proposal oracle only; never deploy oracle masks.",
    "broad_only": broad_values,
    "broad_plus_initial": full_values,
    "broad_plus_initial_minus_broad_only": deltas,
    "retention_gate": "retain initial bank if global, autoPETCT, or LNDb DSC gain is >=0.005",
    "decision": "RETAIN_INITIAL_BANK" if retain else "BROAD_BANK_SUFFICIENT",
    "public_read": False,
    "hidden_read": False,
    "submission_created": False,
}
destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

date -Ins > "${output}/ABLATION_COMPLETE.txt"
cat "${output}/DECISION.json"
