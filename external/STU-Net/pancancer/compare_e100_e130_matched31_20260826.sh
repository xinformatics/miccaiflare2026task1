#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

# Public-only trend diagnostic over the exact 31 epoch-130 cases preserved when
# the full comparison was interrupted. This subset has no submission-selection
# authority; it only asks whether the trained weights are moving usefully.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
analyzer="${root}/analyze_stunet_gf_ensemble.py"
source_root="${root}/public_sweeps/stunetb_matched_e130_20260826"
e130_source="${source_root}/base_predictions"
e100_source="${root}/public_sweeps/stunetb_e100_ensemble_20260826/predictions"
labels_source="${project_root}/validation/Validation-Public-Labels"
anchor_source="${project_root}/external/GF-Screen/analysis/lesion_error_touchcomp_20260825/predictions"
free_source="${project_root}/external/FreeTumor/public_sweeps/independent_baseline_20260825/predictions"
e100_checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0100.pth"
e130_checkpoint="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancer__nnUNetPlans__3d_fullres_STUNetB_PartialBox/fold_all/checkpoint_epoch_0130.pth"
e100_sha="20d94b33e378db1a5d5bd6f612df8fa95b2b97bcc88abb028711374857333675"
e130_sha="90ae4505e0eede40d301f60089931c3ab6967383eb6693e09a251e2952edbf75"
output="${root}/public_sweeps/stunetb_e100_e130_matched31_diag_20260826"
terminal="${output}/TERMINAL.json"

mkdir -p "${output}"/{labels,anchor,freetumor,e100,e130}
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "matched-31 diagnostic already active" >&2; exit 1; }
[[ "$(sha256sum "${e100_checkpoint}" | awk '{print $1}')" == "${e100_sha}" ]]
[[ "$(sha256sum "${e130_checkpoint}" | awk '{print $1}')" == "${e130_sha}" ]]

"${python_bin}" - "${source_root}" "${e130_source}" "${e130_sha}" \
  "${labels_source}" "${anchor_source}" "${free_source}" "${e100_source}" \
  "${output}" <<'PY'
import json
import os
import sys
from pathlib import Path

source_root, e130, expected_sha, labels, anchor, free, e100, output = (
    Path(value) if index != 2 else value
    for index, value in enumerate(sys.argv[1:])
)
terminals = sorted(source_root.glob("BASE_CASE_*.json"))
if len(terminals) != 31:
    raise SystemExit(f"expected 31 preserved terminals, found {len(terminals)}")
cases = []
for path in terminals:
    item = json.loads(path.read_text(encoding="utf-8"))
    if (
        item.get("status") != "COMPLETE"
        or item.get("dataset_role") != "public"
        or item.get("case_count") != 1
        or item.get("checkpoint", {}).get("sha256") != expected_sha
        or item.get("public_labels_read")
        or item.get("hidden_read")
        or item.get("submission_created")
        or item.get("uploaded")
    ):
        raise SystemExit(f"invalid preserved terminal: {path}")
    cases.extend(item["cases"])
if len(cases) != len(set(cases)) != 0:
    raise SystemExit("duplicate preserved case")
if len(cases) != 31:
    raise SystemExit("preserved case coverage mismatch")

for case in sorted(cases):
    probability = case.removesuffix(".nii.gz") + ".prob16.npz"
    sources = {
        "labels": labels / case,
        "anchor": anchor / case,
        "freetumor": free / case,
        "e100_mask": e100 / case,
        "e100_prob": e100 / probability,
        "e130_mask": e130 / case,
        "e130_prob": e130 / probability,
    }
    if not all(path.is_file() for path in sources.values()):
        missing = [str(path) for path in sources.values() if not path.is_file()]
        raise SystemExit(f"missing matched inputs: {missing}")
    links = {
        output / "labels" / case: sources["labels"],
        output / "anchor" / case: sources["anchor"],
        output / "freetumor" / case: sources["freetumor"],
        output / "e100" / case: sources["e100_mask"],
        output / "e100" / probability: sources["e100_prob"],
        output / "e130" / case: sources["e130_mask"],
        output / "e130" / probability: sources["e130_prob"],
    }
    for link, target in links.items():
        if link.is_symlink() or link.exists():
            if link.resolve() != target.resolve():
                raise SystemExit(f"existing link mismatch: {link}")
        else:
            os.symlink(target.resolve(), link)
PY

if [[ ! -s "${output}/e100_analysis.json" ]]; then
  "${python_bin}" -u "${analyzer}" \
    --labels "${output}/labels" --anchor "${output}/anchor" \
    --stunet "${output}/e100" --stunet-probabilities "${output}/e100" \
    --freetumor "${output}/freetumor" --workers 1 \
    --output "${output}/e100_analysis.json" >"${output}/e100_analysis.log" 2>&1
fi
if [[ ! -s "${output}/e130_analysis.json" ]]; then
  "${python_bin}" -u "${analyzer}" \
    --labels "${output}/labels" --anchor "${output}/anchor" \
    --stunet "${output}/e130" --stunet-probabilities "${output}/e130" \
    --freetumor "${output}/freetumor" --workers 1 \
    --output "${output}/e130_analysis.json" >"${output}/e130_analysis.log" 2>&1
fi

"${python_bin}" - "${output}/e100_analysis.json" "${output}/e130_analysis.json" \
  "${e100_checkpoint}" "${e130_checkpoint}" "${terminal}" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

e100_path, e130_path, e100_ckpt, e130_ckpt, output = map(Path, sys.argv[1:])
e100 = json.loads(e100_path.read_text(encoding="utf-8"))
e130 = json.loads(e130_path.read_text(encoding="utf-8"))
if any(item.get("status") != "COMPLETE" or item.get("case_count") != 31 for item in (e100, e130)):
    raise SystemExit("invalid matched analysis")
keys = sorted(set(e100["metrics"]) & set(e130["metrics"]))
deltas = {
    key: {
        metric: e130["metrics"][key][metric] - e100["metrics"][key][metric]
        for metric in ("Lesion_DSC", "Lesion_NSD")
    }
    for key in keys
}
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": "COMPLETE",
    "role": "matched_31_case_checkpoint_trend_diagnostic_only",
    "case_count": 31,
    "selection_authority": False,
    "reason": "preserved processing-order subset is not the fixed public 50",
    "checkpoint_100": {"path": str(e100_ckpt.resolve()), "sha256": sha(e100_ckpt)},
    "checkpoint_130": {"path": str(e130_ckpt.resolve()), "sha256": sha(e130_ckpt)},
    "analysis_100": {"path": str(e100_path.resolve()), "sha256": sha(e100_path)},
    "analysis_130": {"path": str(e130_path.resolve()), "sha256": sha(e130_path)},
    "standalone": {
        "epoch100": e100["metrics"]["stunet_standalone"],
        "epoch130": e130["metrics"]["stunet_standalone"],
        "delta": deltas["stunet_standalone"],
    },
    "anchor_confirmation_either_p70": {
        "epoch100": e100["metrics"]["gf_anchor_confirm_either_p70"],
        "epoch130": e130["metrics"]["gf_anchor_confirm_either_p70"],
        "delta": deltas["gf_anchor_confirm_either_p70"],
    },
    "high_value_public_diagnostic": {
        "epoch100": e100["high_value_public_diagnostic"],
        "epoch130": e130["high_value_public_diagnostic"],
    },
    "all_metric_deltas": deltas,
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
