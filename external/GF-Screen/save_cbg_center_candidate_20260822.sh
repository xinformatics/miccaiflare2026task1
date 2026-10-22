#!/usr/bin/env bash
set -euo pipefail

# Save, never upload, the strongest publicly qualified dense-centerness model.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_glance_cbg_center_deployed_20260822"
small_gate="${audit}/small_lesion_audit/small_lesion_gate.tsv"
output="${root}/public_sweeps/gf_glance_cbg_center_saved_20260822"
materializer="${root}/materialize_deployed_candidate_submission_20260821.sh"
ready="${project}/submissions/READY_20260822.tsv"
submitted="${project}/submissions/codabench_submitted_sha256.tsv"

mkdir -p "${output}"
exec 9>"${root}/runs/save_cbg_center_candidate_20260822.lock"
flock -n 9 || { echo "CBG center saver already active" >&2; exit 1; }
[[ -s "${output}/SAVE_ONLY_COMPLETE.txt" ]] && exit 0
[[ -s "${audit}/PUBLIC_ONLY_COMPLETE.txt" && -s "${audit}/summary.tsv" \
    && -s "${small_gate}" && -s "${audit}/small_lesion_audit/COMPLETE.txt" ]]
queue_floor=-1
if [[ -s "${ready}" && $(awk 'END {print NR-1}' "${ready}") -ge 3 ]]; then
  queue_floor=$(awk -F '\t' 'NR==2{f=($5+$6)/2} NR>2{s=($5+$6)/2;if(s<f)f=s} END{printf "%.9f",f}' "${ready}")
fi
selected=""
selected=$("${python_bin}" - "${small_gate}" "${queue_floor}" <<'PY'
import csv
import json
import sys

path, queue_floor = sys.argv[1], float(sys.argv[2])
with open(path, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
candidates = []
for row in rows:
    if row.get("qualified") != "true":
        continue
    with open(row["robust"], encoding="utf-8") as stream:
        robust = json.load(stream)
    decision = robust["decision"]
    tail_guard = bool(robust.get("checks", {}).get("tail_regression_guard", False))
    combined = float(row["combined"])
    if decision == "PROMOTE" or (tail_guard and combined > queue_floor + 1e-6):
        row["decision"] = decision
        candidates.append(row)
if candidates:
    row = max(candidates, key=lambda item: (float(item["combined"]), float(item["DSC"])))
    fields = (
        "variant", "checkpoint_tag", "checkpoint", "glance_threshold",
        "center_threshold", "DSC", "NSD", "combined", "metrics", "robust",
        "decision",
    )
    print("\t".join(str(row.get(field, "off")) for field in fields))
PY
)
if [[ -z "${selected}" ]]; then
  date -Ins > "${output}/NO_ELIGIBLE_CBG_CENTER_CANDIDATE.txt"
  date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
  exit 0
fi
IFS=$'\t' read -r arm checkpoint_tag checkpoint glance center dsc nsd combined metrics robust decision <<< "${selected}"
[[ "${arm}" == *w025* ]] && family=cen25 || family=cen75
epoch=${checkpoint_tag#checkpoint_epoch_}; epoch=${epoch%%_*}; epoch=$((10#${epoch}))
glance_tag=${glance/./p}
center_tag=${center/./p}
tag="gf_${family}_e${epoch}_g${glance_tag}_c${center_tag}"
materialize_env=(env ALLOW_FALLBACK=1 CHECKPOINT="${checkpoint}" TAG="${tag}" \
  METRICS="${metrics}" ROBUST_JSON="${robust}" GLANCE_THRESHOLD="${glance}" \
  LESION_THRESHOLD=0.50 GLANCE_CENTER_LOCALIZATION=1)
[[ "${center}" != off ]] && materialize_env+=(GLANCE_CENTER_THRESHOLD="${center}")
"${materialize_env[@]}" bash "${materializer}"
submission="${project}/submissions/${tag}.zip"
unzip -tq "${submission}"
[[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
digest=$(sha256sum "${submission}" | awk '{print $1}')
saved="${output}/saved_cbg_center_submission.tsv"
printf 'rank\ttag\tfamily\tcheckpoint\tglance_threshold\tcenter_threshold\tlesion_threshold\tDSC\tNSD\tmetrics\trobust\tgate\tsubmission\tsha256\n' > "${saved}"
printf '1\t%s\t%s\t%s\t%s\t%s\t0.50\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${tag}" "${family}" "${checkpoint}" "${glance}" "${center}" "${dsc}" "${nsd}" \
  "${metrics}" "${robust}" "${decision}" "${submission}" "${digest}" >> "${saved}"

"${python_bin}" "${root}/refresh_all_saved_candidates_20260822.py" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/saved_backup_submissions.tsv" \
  --manifest "${root}/public_sweeps/gf_v7_appearance_saved_20260822/saved_appearance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_component_bg_saved_20260822/saved_glance_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_cbg_intersection_saved_20260822/saved_intersection_submission.tsv" \
  --manifest "${root}/public_sweeps/a055_a069_bbox_wise_saved_20260822/saved_wise_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_boxloc_saved_20260822/saved_boxloc_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_glance_cbg_box_saved_20260822/saved_cbg_box_submission.tsv" \
  --manifest "${root}/public_sweeps/gf_focus_allpartial_saved_20260822/saved_focus_submission.tsv" \
  --manifest "${root}/public_sweeps/qualified_focus_glance_saved_20260822/saved_composite_submission.tsv" \
  --manifest "${saved}" --submitted "${submitted}" --output "${ready}" \
  --audit-output "${project}/submissions/READY_20260822.audit.json" > "${output}/refresh_ready.log"
"${python_bin}" "${root}/rank_submission_candidates_calibrated.py" \
  --history "${root}/codabench_history.tsv" --candidates "${ready}" \
  --output "${project}/submissions/READY_20260822.calibrated.tsv" \
  --json-output "${project}/submissions/READY_20260822.calibrated.json" \
  > "${output}/calibrated_rank.log"
date -Ins > "${output}/SAVE_ONLY_COMPLETE.txt"
