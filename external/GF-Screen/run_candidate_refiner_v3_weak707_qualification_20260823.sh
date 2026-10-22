#!/usr/bin/env bash
set -euo pipefail

# Deployment-aware exact-707 qualification. Standalone selections compare
# against the strongest exact-210 OOF baseline on the shared 210-case subset;
# anchor selections use their paired exact-707 fixed-currentgf composition
# gate. No raw label, public, hidden, packaging, submission, or upload path exists.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
scaled_root="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
scaled_complete="${scaled_root}/FULL_VOLUME_OOF_COMPLETE.json"
scaled_skipped="${scaled_root}/FULL_VOLUME_OOF_SKIPPED.json"
output="${root}/analysis/candidate_refiner_v4_cell707_qualification_20260823"
prepared="${output}/PREPARED.json"
robust="${output}/scaled_vs_best210.robust.json"
complete="${output}/QUALIFICATION.json"

primary="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823"
union="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
weak_transfer="${root}/analysis/candidate_refiner_v2_weak_transfer_oof_20260823"

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v3_weak707_qualification_20260823.lock"
flock -n 9 || { echo "weak707 OOF qualification already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${scaled_complete}" && ! -s "${scaled_skipped}" ]]; do
  printf '%s waiting for exact-707 full-volume OOF\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${scaled_skipped}" ]]; then
  "${python_bin}" - "${scaled_skipped}" "${complete}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
output.write_text(json.dumps({
  "created_at":datetime.now().astimezone().isoformat(),
  "decision":"SKIP_FIXED_PUBLIC",
  "reason":"exact-707 OOF was skipped",
  "source":json.loads(source.read_text(encoding="utf-8")),
},indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi

while [[ ! -s "${primary}/FULL_VOLUME_OOF_COMPLETE.txt" \
    || ! -s "${union}/FULL_VOLUME_OOF_COMPLETE.txt" \
    || ! -s "${weak_transfer}/TRANSFER_OOF_COMPLETE.txt" ]]; do
  printf '%s waiting for all completed exact-210 OOF baselines\n' "$(date -Ins)"
  sleep 30
done

"${python_bin}" -u "${root}/prepare_scaled_oof_comparison.py" \
  --subset-manifest "${manifest}" \
  --scaled-calibration "${scaled_root}/deployment_calibration.json" \
  --baseline "primary=${primary}/deployment_calibration.json" \
  --baseline "union=${union}/deployment_calibration.json" \
  --baseline "weak_transfer=${weak_transfer}/deployment_calibration.json" \
  --output "${output}" > "${output}/prepare.log" 2>&1

anchor_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["baseline_metrics"])' "${prepared}")
candidate_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["scaled_subset_metrics"])' "${prepared}")
anchor_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["selected_baseline"])' "${prepared}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name "${anchor_name}" --candidate-name cell707_scaled_subset \
  --output "${robust}" --bootstrap 20000 --seed 20260823 \
  --min-dsc-gain 0.005 --min-nsd-gain 0.005 --min-win-rate 0.52 \
  --max-catastrophic-rate 0.10 > "${output}/robust.log" 2>&1

"${python_bin}" - "${prepared}" "${robust}" "${complete}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
prepared_path,robust_path,output=map(Path,sys.argv[1:])
prepared=json.loads(prepared_path.read_text(encoding="utf-8"))
robust=json.loads(robust_path.read_text(encoding="utf-8"))
scaled_path=Path(prepared["scaled_calibration"])
scaled=json.loads(scaled_path.read_text(encoding="utf-8"))
deployment=prepared["scaled_selected_deployment"]
if deployment != scaled.get("calibration_deployment"):
    raise SystemExit("prepared/scaled deployment mismatch")
if deployment == "anchor_currentgf":
    composition=scaled.get("composition_gate_payload",{})
    checks=composition.get("checks",{})
    qualified=(composition.get("decision")=="PROMOTE" and checks
               and all(value is True for value in checks.values()))
    qualification_path="paired_exact707_anchor_composition"
elif deployment == "standalone":
    composition=scaled.get("composition_gate_payload")
    qualified=robust.get("decision")=="PROMOTE"
    qualification_path="matched_shared210_standalone_scale"
else:
    raise SystemExit(f"invalid selected deployment: {deployment}")
payload={
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "selection_basis":"deployment-aware exact-707 OOF qualification before a single fixed-public test",
  "qualification_path":qualification_path,
  "decision":"QUALIFY_FOR_FIXED_PUBLIC" if qualified else "KEEP_OOF_BASELINE",
  "prepared":str(prepared_path.resolve()),
  "paired_comparison":str(robust_path.resolve()),
  "paired_comparison_payload":robust,
  "composition_gate_payload":composition,
  "selected_baseline":prepared["selected_baseline"],
  "scaled_selected_deployment":prepared["scaled_selected_deployment"],
  "scaled_comparison_deployment":prepared["scaled_comparison_deployment"],
  "scaled_comparison_calibration":prepared["scaled_comparison_calibration"],
  "scaled_deployment_mode":prepared["scaled_deployment_mode"],
  "public_read":False,
  "hidden_read":False,
  "submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload,indent=2))
PY
