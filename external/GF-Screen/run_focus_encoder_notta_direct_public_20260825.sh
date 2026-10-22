#!/usr/bin/env bash
set -euo pipefail

# Fixed-public-only evaluation of the strongest untested trained-model route.
# B32 handles mature proposals, frozen dual-window Focus handles weak sources,
# and the encoder-adapted Focus handles lung cells. TTA is deliberately absent.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
teacher_work="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
teacher_release="${teacher_work}/TEACHER_INFERENCE_COMPLETE.json"
teacher_checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
teacher_audit="${teacher_work}/train_unlabel_audit.json"
frozen="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_epoch_routes.json"
encoder="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824/focus_epoch_routes.json"
upgrade="${root}/analysis/focus_dualwindow_encoder01b8_upgrade_fivefold_20260824/RESTRICTED_DECISION.json"
mixed="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823/focus_b32_source_hybrid_decision.json"
base_calibration="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823/deployment_calibration.json"
anchor_metrics="${root}/analysis/lesion_error_touchcomp_20260825/fixed_public.metrics.csv"
work="${root}/public_sweeps/focus_encoder_notta_direct_20260825"
route="${work}/ROUTE.json"
qualification="${work}/PUBLIC_ELIGIBILITY.json"
calibration="${work}/PREDECLARED_CALIBRATION.json"
terminal="${work}/TERMINAL.json"
lock="${root}/runs/run_focus_encoder_notta_direct_public_20260825.lock"

mkdir -p "${work}" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "encoder Focus no-TTA public screen already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

while [[ ! -s "${teacher_release}" ]]; do
  printf '%s waiting for exact Train-Unlabel teacher inventories to release GPUs\n' "$(date -Ins)"
  sleep 30
done
for required in "${teacher_checkpoint}" "${teacher_audit}" "${frozen}" \
    "${encoder}" "${upgrade}" "${mixed}" "${base_calibration}" \
    "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing route prerequisite: ${required}" >&2; exit 2; }
done

"${python_bin}" - "${teacher_release}" "${teacher_checkpoint}" "${teacher_audit}" <<'PY'
import hashlib,json,sys
from pathlib import Path
terminal,checkpoint,audit=map(Path,sys.argv[1:]);payload=json.load(open(terminal,encoding="utf-8"));sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if payload.get("status")!="COMPLETE" or payload.get("decision")!="TEACHER_INFERENCE_COMPLETE_GPU_RELEASE_PENDING": raise SystemExit("invalid teacher release terminal")
if payload.get("source_inventory")!=2856 or payload.get("usable_after_exact_overlap_exclusion")!=2850: raise SystemExit("teacher release inventory mismatch")
if set(payload.get("threshold_inventories",{}))!={"0.85","0.90","0.95","0.97"} or any(item.get("members")!=2856 for item in payload["threshold_inventories"].values()): raise SystemExit("teacher threshold inventory mismatch")
for key,path in (("teacher_checkpoint",checkpoint),("source_audit",audit)):
    identity=payload.get(key,{})
    if Path(identity.get("path","")).resolve()!=path.resolve() or identity.get("sha256")!=sha(path): raise SystemExit(f"teacher release {key} identity mismatch")
if payload.get("pseudo_materialization_complete") is not False or payload.get("public_read") is not False or payload.get("hidden_read") is not False or payload.get("submission_created") is not False or payload.get("uploaded") is not False: raise SystemExit("teacher release provenance mismatch")
PY

"${python_bin}" "${root}/compose_focus_encoder_notta_routes.py" \
  --frozen-routes "${frozen}" --encoder-routes "${encoder}" \
  --encoder-decision "${upgrade}" --mixed-decision "${mixed}" \
  --output "${route}" > "${work}/compose.log"

"${python_bin}" - "${route}" "${base_calibration}" "${anchor_metrics}" \
  "${qualification}" "${calibration}" <<'PY'
import csv, datetime, hashlib, json, math, os, sys
from pathlib import Path
route_path, base_path, anchor_path, qualification_path, calibration_path = map(Path, sys.argv[1:])
route = json.load(open(route_path, encoding="utf-8"))
base = json.load(open(base_path, encoding="utf-8"))
if route.get("decision") != "QUALIFY_ENCODER_FOCUS_B32_NOTTA_FOR_FIXED_PUBLIC":
    raise SystemExit("route identity drift")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
anchor_rows = list(csv.DictReader(open(anchor_path, encoding="utf-8")))
if len(anchor_rows) != 50 or not all(
    {"Name", "Lesion_DSC", "Lesion_NSD"} <= set(row) for row in anchor_rows
):
    raise SystemExit("fixed-public touchcomp anchor inventory mismatch")
anchor_dsc = math.fsum(float(row["Lesion_DSC"]) for row in anchor_rows) / 50
anchor_nsd = math.fsum(float(row["Lesion_NSD"]) for row in anchor_rows) / 50
if not math.isclose(anchor_dsc, 0.6004316379500749, rel_tol=0, abs_tol=1e-12):
    raise SystemExit("fixed-public touchcomp DSC identity mismatch")
if not math.isclose(anchor_nsd, 0.5104225107769019, rel_tol=0, abs_tol=1e-12):
    raise SystemExit("fixed-public touchcomp NSD identity mismatch")
now = datetime.datetime.now().astimezone().isoformat()
qualification = {
    "schema_version": 1,
    "created_at": now,
    "decision": "PROMOTE",
    "meaning": "eligible for one fixed-public trained-model screen only; partial labels are not complete-negative evidence",
    "route": str(route_path.resolve()),
    "route_sha256": sha(route_path),
    "fixed_public_anchor": {
        "name": "gf_touchcomp_0825",
        "path": str(anchor_path.resolve()),
        "sha256": sha(anchor_path),
        "cases": len(anchor_rows),
        "Lesion_DSC": anchor_dsc,
        "Lesion_NSD": anchor_nsd,
    },
    "public_read": False,
    "hidden_read": False,
    "submission_created": False,
}
calibration = dict(base)
calibration.update({
    "schema_version": int(base.get("schema_version", 1)),
    "created_at": now,
    "selection_basis": "pre-public B32 deployment calibration plus route-specific known-positive contour thresholds",
    "source_segmentation_thresholds": route["source_segmentation_thresholds"],
    "route": str(route_path.resolve()),
    "route_sha256": sha(route_path),
    "public_read": False,
    "hidden_read": False,
})
for path, payload in ((qualification_path, qualification), (calibration_path, calibration)):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
PY

WEAK707_PUBLIC_QUALIFICATION="${qualification}" \
WEAK707_PUBLIC_REQUIRED_DECISION=PROMOTE \
WEAK707_PUBLIC_OOF_ROOT="${work}" \
WEAK707_PUBLIC_CALIBRATION="${calibration}" \
WEAK707_PUBLIC_ANCHOR_METRICS="${anchor_metrics}" \
WEAK707_PUBLIC_COMPOSITE_ROUTE_DECISION="${route}" \
WEAK707_PUBLIC_OUTPUT="${work}" \
WEAK707_PUBLIC_PROPOSAL_CACHE_ROOT="${root}/public_sweeps/candidate_refiner_v4_cell707_20260823" \
WEAK707_PUBLIC_TAG_PREFIX=fen25 \
WEAK707_PUBLIC_LOCK="${root}/runs/run_focus_encoder_notta_direct_public_inner_20260825.lock" \
WEAK707_PUBLIC_GPU0_MAX_MIB=4096 \
WEAK707_PUBLIC_GPU1_MAX_MIB=1024 \
bash "${root}/run_candidate_refiner_v3_weak707_public_20260823.sh" \
  >> "${work}/controller.log" 2>&1

summary="${work}/summary_by_dsc.tsv"
[[ -s "${summary}" ]] || { echo "missing fixed-public summary" >&2; exit 3; }
read -r candidate_name candidate_metrics < <("${python_bin}" - "${summary}" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("expected exactly one fixed-public candidate")
print(rows[0]["variant"], rows[0]["metrics"], sep="\t")
PY
)
[[ -s "${candidate_metrics}" ]] || { echo "missing direct candidate metrics" >&2; exit 3; }
candidate_predictions="${work}/${candidate_name}"
[[ $(find "${candidate_predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "direct candidate prediction inventory mismatch" >&2; exit 3;
}
lesion_error="${work}/lesion_error_${candidate_name}"
if [[ ! -s "${lesion_error}/summary.json" ]]; then
  "${python_bin}" -u "${root}/analyze_lesion_errors.py" \
    --labels "$(cd "${root}/../.." && pwd)/validation/Validation-Public-Labels" \
    --predictions "${candidate_predictions}" --output "${lesion_error}" \
    > "${work}/lesion_error_${candidate_name}.log" 2>&1
fi
[[ -s "${lesion_error}/summary.json" ]]
paired_gate="${work}/TOUCHCOMP_DSC_FIRST_GATE.json"
"${python_bin}" -u "${root}/select_dsc_first_nonregression.py" \
  --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name gf_touchcomp_0825 --candidate-name "${candidate_name}" \
  --min-dsc-gain 0.015 --min-dsc-win-rate 0.55 --min-nsd-gain 0.0 \
  --min-nsd-uncertainty-bound -0.005 --bootstrap 20000 --seed 20260825 \
  --output "${paired_gate}" > "${work}/touchcomp_dsc_first_gate.log"

"${python_bin}" - "${summary}" "${route}" "${calibration}" \
  "${anchor_metrics}" "${paired_gate}" "${candidate_predictions}" \
  "${lesion_error}/summary.json" "${terminal}" <<'PY'
import csv, datetime, hashlib, json, os, sys
from pathlib import Path
summary, route, calibration, anchor, paired_gate, candidate_predictions, lesion_error, output = map(Path, sys.argv[1:])
rows = list(csv.DictReader(open(summary, encoding="utf-8"), delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("expected exactly one fixed-public candidate")
row = rows[0]
summary_dsc = float(row["DSC"])
summary_nsd = float(row["NSD"])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
gate = json.load(open(paired_gate, encoding="utf-8"))
candidate_metrics = Path(row["metrics"])
diagnostic = json.load(open(lesion_error, encoding="utf-8"))
if (
    int(diagnostic.get("cases", -1)) != 50
    or Path(str(diagnostic.get("configuration", {}).get("predictions", ""))).resolve()
    != candidate_predictions.resolve()
):
    raise SystemExit("direct lesion-error diagnostic identity mismatch")
if (
    gate.get("anchor") != "gf_touchcomp_0825"
    or Path(gate.get("anchor_metrics", "")).resolve() != anchor.resolve()
    or gate.get("anchor_metrics_sha256") != sha(anchor)
    or Path(gate.get("candidate_metrics", "")).resolve() != candidate_metrics.resolve()
    or gate.get("candidate_metrics_sha256") != sha(candidate_metrics)
    or int(gate.get("cases", -1)) != 50
):
    raise SystemExit("direct paired-gate identity mismatch")
criteria = gate.get("criteria", {})
expected_criteria = {
    "bootstrap": 20000,
    "seed": 20260825,
    "min_dsc_gain": 0.015,
    "min_dsc_win_rate": 0.55,
    "min_nsd_gain": 0.0,
    "min_nsd_uncertainty_bound": -0.005,
    "max_catastrophic_dsc_rate": 0.10,
    "catastrophic_dsc_delta": -0.20,
}
if criteria != expected_criteria:
    raise SystemExit("direct paired-gate criteria mismatch")
candidate_mean = gate.get("candidate_mean", {})
exact_dsc = float(candidate_mean.get("Lesion_DSC", -1))
exact_nsd = float(candidate_mean.get("Lesion_NSD", -1))
if (
    abs(exact_dsc - summary_dsc) > 5.1e-5
    or abs(exact_nsd - summary_nsd) > 5.1e-5
):
    raise SystemExit("direct paired-gate score mismatch")
dsc = exact_dsc
nsd = exact_nsd
passed = (
    dsc >= 0.6154316379500749
    and nsd >= 0.5104225107769019
    and gate.get("decision") == "PROMOTE"
)
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "decision": "QUALIFY_SUBMISSION_PREPARATION" if passed else "KEEP_TOUCHCOMP",
    "fixed_public": {"DSC": dsc, "NSD": nsd, "DSC_gate": 0.6154316379500749, "NSD_floor": 0.5104225107769019},
    "fixed_public_anchor": {"path": str(anchor.resolve()), "sha256": sha(anchor)},
    "paired_gate": {"path": str(paired_gate.resolve()), "sha256": sha(paired_gate), "decision": gate["decision"]},
    "lesion_error_diagnostic": {
        "summary": {"path": str(lesion_error.resolve()), "sha256": sha(lesion_error)},
        "selection_role": "diagnostic_only_not_used_for_promotion",
    },
    "route": {"path": str(route.resolve()), "sha256": sha(route)},
    "calibration": {"path": str(calibration.resolve()), "sha256": sha(calibration)},
    "summary": {"path": str(summary.resolve()), "sha256": sha(summary)},
    "public_read": True,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
cat "${summary}"
