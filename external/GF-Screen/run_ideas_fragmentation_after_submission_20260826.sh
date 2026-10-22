#!/usr/bin/env bash
set -euo pipefail

# Public-only forensic check from ideas.txt.  It waits for the qualified
# hidden archive to finish so this CPU/RAM work cannot delay that handoff.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
saver_terminal="${project_root}/external/STU-Net/pancancer/public_sweeps/stunetb_matched_e130_saved_20260826/TERMINAL.json"
raw="${root}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/inference"
historical="${root}/public_sweeps/gf_glance_component_bg_deployed_20260822/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283/g0p20/filtered_mm1000"
anchor="${root}/analysis/lesion_error_touchcomp_20260825/predictions"
labels="${project_root}/validation/Validation-Public-Labels"
audit_root="${root}/analysis/fragmentation_connectivity_20260826"
candidate_root="${root}/public_sweeps/cbg_e283_g020_connectivity26_20260826"

mkdir -p "${audit_root}" "${candidate_root}"
while :; do
  if [[ -s "${saver_terminal}" ]]; then
    status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${saver_terminal}")
    [[ "${status}" == COMPLETE ]] && break
    [[ "${status}" == FAILED ]] && { echo "qualified saver failed; preserving resources for diagnosis" >&2; exit 1; }
  fi
  printf '%s waiting for qualified hidden archive before public fragmentation audit\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" -u "${root}/audit_fragmentation_connectivity.py" \
  --raw-dir "${raw}" \
  --historical-filtered-dir "${historical}" \
  --anchor-dir "${anchor}" \
  --labels-dir "${labels}" \
  --minimum-mm3 1000 \
  --large-mm3 3000 \
  --detection-overlap 0.1 \
  --output "${audit_root}/audit.json" \
  > "${audit_root}/audit.log" 2>&1

"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${raw}" \
  --fusion a \
  --min-mm3 1000 \
  --connectivity 26 \
  --workers 1 \
  --output "${candidate_root}/filtered_mm1000" \
  > "${candidate_root}/filter.log" 2>&1

"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${candidate_root}/filtered_mm1000" \
  --gt "${labels}" \
  --csv "${candidate_root}/metrics.csv" \
  --workers 1 \
  > "${candidate_root}/score.log" 2>&1

"${python_bin}" - "${audit_root}/audit.json" "${candidate_root}/metrics.csv" \
  "${candidate_root}/TERMINAL.json" <<'PY'
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

audit_path, metrics_path, output = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
if len(rows) != 50 or audit.get("status") != "COMPLETE":
    raise SystemExit("incomplete public-only connectivity audit")
sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "role": "fixed_public_connectivity_diagnostic_only",
    "connectivity": 26,
    "minimum_component_mm3": 1000,
    "mean_DSC": sum(float(row["Lesion_DSC"]) for row in rows) / 50,
    "mean_NSD": sum(float(row["Lesion_NSD"]) for row in rows) / 50,
    "fragmentation_audit": {
        "path": str(audit_path.resolve()),
        "sha256": sha256(audit_path),
        "bbox_fragmentation_possible": audit["bbox_fragmentation_possible"],
        "connectivity26_strict_recoveries": audit["connectivity26_strict_recoveries"],
    },
    "metrics": {"path": str(metrics_path.resolve()), "sha256": sha256(metrics_path)},
    "public_labels_read": True,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
print(json.dumps(payload, indent=2))
PY
