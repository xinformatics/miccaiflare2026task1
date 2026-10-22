#!/usr/bin/env bash
set -euo pipefail

# Post-selection diagnostic on exact-707 excluded-fold predictions. It waits
# until the first primary save-or-skip terminal so it cannot delay tonight's
# submission candidate. No public/hidden path or packaging is reachable.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
oof="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
complete="${oof}/FULL_VOLUME_OOF_COMPLETE.json"
calibration="${oof}/deployment_calibration.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
labels="${oof}/staging/all/labels"
save_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
output="${root}/analysis/candidate_refiner_v4_cell707_failure_modes_20260824"
terminal="${output}/FAILURE_MODE_AUDIT.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v4_exact707_failure_audit_20260824.lock"
flock -n 9 || { echo "exact-707 failure audit already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${complete}" || ! -s "${save_terminal}" ]]; do
  printf '%s waiting for exact-707 OOF and primary save-or-skip terminal\n' "$(date -Ins)"
  sleep 30
done
for required in "${calibration}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing required audit input: ${required}" >&2; exit 3; }
done
read -r deployment variant < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
deployment=p.get("calibration_deployment","standalone")
variant=str(p.get("variant",""))
if deployment not in {"standalone","anchor_currentgf"} or not variant or "/" in variant:
 raise SystemExit("invalid frozen deployment calibration")
print(deployment,variant)
PY
)
if [[ "${deployment}" == anchor_currentgf ]]; then
  predictions="${oof}/anchor_currentgf/${variant}/combined"
else
  predictions="${oof}/${variant}/combined"
fi
"${python_bin}" - "${manifest}" "${labels}" "${predictions}" <<'PY'
import json,sys
from pathlib import Path
manifest,labels,predictions=map(Path,sys.argv[1:])
expected={item["prediction_name"] for item in json.loads(manifest.read_text())["items"]}
observed_labels={path.name for path in labels.glob("*.nii.gz")}
observed_predictions={path.name for path in predictions.glob("*.nii.gz")}
if len(expected)!=707 or observed_labels!=expected or observed_predictions!=expected:
 raise SystemExit("exact-707 audit coverage mismatch")
PY

nice -n 10 "${python_bin}" -u "${root}/analyze_lesion_errors.py" \
  --labels "${labels}" --predictions "${predictions}" \
  --output "${output}/component_errors" \
  > "${output}/component_errors.log" 2>&1
"${python_bin}" -u "${root}/summarize_candidate_refiner_failure_modes.py" \
  --manifest "${manifest}" --cases "${output}/component_errors/cases.csv" \
  --lesion-summary "${output}/component_errors/summary.json" \
  --calibration "${calibration}" --output "${terminal}" \
  > "${output}/summarize.log" 2>&1
cat "${terminal}"
