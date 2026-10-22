#!/usr/bin/env bash
set -euo pipefail

# Same-day, label-blind checkpoint-293 consumer. The public evaluator is a
# separate service. This script waits for its hash-bound terminal, creates one
# hidden archive only after the fixed promotion gate passes, and always
# recreates the interrupted long-Focus controller from its latest checkpoint.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0293.pt"
resume_checkpoint="${root}/runs/gf_focus_pu_long30_e283_to313_20260825/checkpoint_epoch_0297.pt"
public="${root}/public_sweeps/gf_focus_pu_long30_e293_interim_20260825"
public_terminal="${public}/TERMINAL.json"
summary="${public}/summary.tsv"
hidden_images="${project}/validation/Validation-Hidden-Images"
prompt_support="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/hidden_biomedparse_guided_whole_body_lesion_e050_v500_l0.50_mm500/raw"
gfconf_archive="${project}/submissions/gf_conf_cbg41_e283.zip"
gfconf_sha256="d8f5f18e860ffd3d0a1dfed44112f0983258320c14f11bf64ab5c1ffd6227e4a"
work="${root}/public_sweeps/gf_focus_pu_e293_interim_saved_20260825"
hidden="${root}/hidden_predictions/gf_focus_pu_e293_interim_20260825"
terminal="${work}/TERMINAL.json"
hidden_started="${work}/HIDDEN_INFERENCE_STARTED.json"
lock="${root}/runs/run_interim_e293_predict_and_resume_20260825.lock"
resume_unit="miccai-long-focus-pu-resume-e297-20260825"

mkdir -p "${work}" "${hidden}" "${project}/submissions" "${root}/runs"
exec 9>"${lock}"
flock -n 9 || { echo "interim checkpoint-293 consumer already active" >&2; exit 1; }

resume_training() {
  if systemctl --user is-active --quiet "${resume_unit}.service"; then
    return 0
  fi
  systemd-run --user --unit="${resume_unit}" --collect \
    --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
    /usr/bin/bash "${root}/run_long_focus_pu_after_direct_public_20260825.sh"
}

finalize() {
  local status=$?
  trap - EXIT
  resume_training || true
  exit "${status}"
}
trap finalize EXIT

write_terminal() {
  local status="$1" decision="$2" reason="$3" variant="${4:-}" archive="${5:-}"
  "${python_bin}" - "${status}" "${decision}" "${reason}" "${variant}" \
    "${checkpoint}" "${resume_checkpoint}" "${public_terminal}" "${summary}" \
    "${hidden_started}" "${archive}" "${terminal}" <<'PY'
import datetime, hashlib, json, os, sys, zipfile
from pathlib import Path
status, decision, reason, variant = sys.argv[1:5]
checkpoint, resume_checkpoint, public, summary, hidden_started = map(Path, sys.argv[5:10])
archive = Path(sys.argv[10]) if sys.argv[10] else None
output = Path(sys.argv[11])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
def asset(path):
    return {"path": str(path.resolve()), "sha256": sha(path)} if path.is_file() else None
submission = None
if archive is not None:
    if not archive.is_file() or len(archive.name) >= 64:
        raise SystemExit("archive identity invalid")
    with zipfile.ZipFile(archive) as handle:
        members = [item for item in handle.infolist() if not item.is_dir()]
        if len(members) != 100 or len({item.filename for item in members}) != 100:
            raise SystemExit("archive coverage mismatch")
        if any("/" in item.filename or not item.filename.endswith(".nii.gz") or item.file_size < 1 for item in members):
            raise SystemExit("archive is not an exact flat nonempty NIfTI set")
        if handle.testzip() is not None:
            raise SystemExit("archive CRC failure")
    submission = {"path": str(archive.resolve()), "sha256": sha(archive), "members": 100}
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "status": status,
    "decision": decision,
    "reason": reason,
    "selected_epoch": 293,
    "selected_variant": variant or None,
    "checkpoint": asset(checkpoint),
    "resume_checkpoint": asset(resume_checkpoint),
    "public_terminal": asset(public),
    "public_summary": asset(summary),
    "hidden_inference_started": asset(hidden_started),
    "submission": submission,
    "fixed_public_DSC_gate": 0.6154316379500749,
    "fixed_public_NSD_floor": 0.5104225107769019,
    "hidden_labels_read": False,
    "uploaded": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
}

if [[ -s "${terminal}" ]]; then
  existing_status=$("${python_bin}" -c \
    'import json,sys;print(json.load(open(sys.argv[1],encoding="utf-8"))["status"])' \
    "${terminal}")
  case "${existing_status}" in
    COMPLETE|SKIPPED) exit 0 ;;
    FAILED)
      mv --no-clobber "${terminal}" \
        "${work}/TERMINAL.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected interim terminal status: ${existing_status}" >&2; exit 2 ;;
  esac
fi

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for checkpoint-293 fixed-public terminal\n' "$(date -Ins)"
  sleep 20
done

read -r public_status public_decision variant minimum dsc nsd < <(
  "${python_bin}" - "${public_terminal}" "${summary}" "${checkpoint}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
terminal_path, summary_path, checkpoint = map(Path, sys.argv[1:])
terminal = json.load(open(terminal_path, encoding="utf-8"))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for key, path in (("checkpoint", checkpoint), ("summary", summary_path)):
    identity = terminal.get(key) or {}
    if Path(identity.get("path", "")).resolve() != path.resolve() or identity.get("sha256") != sha(path):
        raise SystemExit(f"public {key} identity mismatch")
status = terminal.get("status")
decision = terminal.get("decision")
variant = terminal.get("selected_variant") or "NONE"
if status != "COMPLETE":
    print(status, decision or "FAILED", "NONE", 0, 0, 0)
    raise SystemExit(0)
rows = list(csv.DictReader(open(summary_path, encoding="utf-8"), delimiter="\t"))
selected = [row for row in rows if row["variant"] == variant]
if decision == "QUALIFY_SUBMISSION_PREPARATION":
    if variant not in {"deployed500", "deployed1000", "touching"} or len(selected) != 1:
        raise SystemExit("invalid public selection")
    row = selected[0]
    dsc, nsd = float(row["DSC"]), float(row["NSD"])
    if row["gate"] != "PROMOTE" or dsc < 0.6154316379500749 or nsd < 0.5104225107769019:
        raise SystemExit("selected row missed the promotion gate")
    minimum = 500 if variant == "deployed500" else 1000
    print(status, decision, variant, minimum, dsc, nsd)
else:
    if decision != "KEEP_TOUCHCOMP" or variant != "NONE":
        raise SystemExit("unexpected non-promotion decision")
    print(status, decision, "NONE", 0, 0, 0)
PY
)

if [[ "${public_status}" != COMPLETE ]]; then
  write_terminal FAILED NO_ARCHIVE "checkpoint-293 public evaluation failed"
  exit 2
fi
if [[ "${public_decision}" != QUALIFY_SUBMISSION_PREPARATION ]]; then
  write_terminal SKIPPED KEEP_TOUCHCOMP \
    "checkpoint 293 did not clear the fixed-public promotion gate"
  exit 0
fi

for directory in "${hidden_images}" "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
    echo "expected exact 100 hidden assets: ${directory}" >&2; exit 2;
  }
done
for file in "${checkpoint}" "${resume_checkpoint}" "${prompt_config}"; do
  [[ -s "${file}" ]] || { echo "missing interim asset: ${file}" >&2; exit 2; }
done

"${python_bin}" - "${checkpoint}" "${variant}" "${hidden_started}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
checkpoint, variant, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "selected_epoch": 293,
    "selected_variant": variant,
    "checkpoint": {"path": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
    "hidden_labels_read": False,
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

mkdir -p "${hidden}/raw" "${hidden}/filtered_mm${minimum}" \
  "${hidden}/prompted_mm${minimum}" "${hidden}/deployed_mm${minimum}"
CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/val_GF_FLARE.py" \
  --test_data_path "${hidden_images}" --save_prediction_path "${hidden}/raw" \
  --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
  --glance_threshold 0.20 --lesion_threshold 0.50 \
  --focus_batch_size 16 --workers 2 --skip_existing \
  > "${hidden}/inference.log" 2>&1
[[ $(find "${hidden}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

filtered="${hidden}/filtered_mm${minimum}"
prompted="${hidden}/prompted_mm${minimum}"
deployed="${hidden}/deployed_mm${minimum}"
"${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
  --pred-a "${hidden}/raw" --fusion a --min-mm3 "${minimum}" --workers 4 \
  --output "${filtered}" > "${hidden}/filter.log" 2>&1
"${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
  --base "${filtered}" --support "${prompt_support}" \
  --config "${prompt_config}" --output "${prompted}" \
  > "${hidden}/prompt.log" 2>&1
"${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
  --base "${prompted}" --candidate "${guided_candidate}" \
  --support "${prompt_support}" --output "${deployed}" \
  --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
  --min-support-fraction 0.25 --min-support-mm3 100 \
  --support-dilation-mm inf --workers 4 > "${hidden}/guided.log" 2>&1
selected="${deployed}"
if [[ "${variant}" == touching ]]; then
  [[ $(sha256sum "${gfconf_archive}" | awk '{print $1}') == "${gfconf_sha256}" ]]
  mkdir -p "${hidden}/gfconf_anchor" "${hidden}/touching"
  if [[ $(find "${hidden}/gfconf_anchor" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne 100 ]]; then
    unzip -oq "${gfconf_archive}" -d "${hidden}/gfconf_anchor"
  fi
  "${python_bin}" -u "${root}/screen_anchor_support_contour_growth.py" materialize \
    --anchor "${hidden}/gfconf_anchor" --support "${deployed}" \
    --configuration touching_component --output "${hidden}/touching" --workers 4 \
    > "${hidden}/touching.log" 2>&1
  selected="${hidden}/touching"
fi
[[ $(find "${selected}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]]

case "${variant}" in
  deployed500) tag="gf_pu293_d500_0825" ;;
  deployed1000) tag="gf_pu293_d1k_0825" ;;
  touching) tag="gf_pu293_touch_0825" ;;
  *) exit 2 ;;
esac
archive="${project}/submissions/${tag}.zip"
"${python_bin}" -u "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${selected}" \
  --output "${archive}" --expected-count 100 > "${hidden}/package.log" 2>&1
sha256sum "${archive}" > "${archive}.sha256"
write_terminal COMPLETE ARCHIVE_READY_FOR_MANUAL_UPLOAD \
  "checkpoint-293 public-qualified deployment reproduced label-blind on 100 hidden images" \
  "${variant}" "${archive}"
printf 'checkpoint\tvariant\tpublic_DSC\tpublic_NSD\tarchive\n293\t%s\t%s\t%s\t%s\n' \
  "${variant}" "${dsc}" "${nsd}" "${archive}" > "${work}/saved_submission.tsv"
