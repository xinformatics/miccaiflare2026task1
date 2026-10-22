#!/usr/bin/env bash
set -euo pipefail

# Label-blind scale-up of the exact current-GF proposal stream used by the
# 210-case high-recall oracle. Reuse the audited 210-case pilot by hard link
# and infer only the remaining 497 complete-label training cases. This script
# never reads labels, public/hidden cases, or submission artifacts.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
pilot_manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
images="${root}/analysis/fpr_training_mining_707_20260822/staged_images"
pilot="${root}/analysis/candidate_refiner_v1_training_20260822/gf_raw_l050"
checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
weak_transfer="${root}/analysis/candidate_refiner_v2_weak_transfer_oof_20260823/TRANSFER_OOF_COMPLETE.txt"
epoch105_complete="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050/PROPOSALS_COMPLETE.json"
output="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
marker="${output}/PROPOSALS_COMPLETE.json"

mkdir -p "${output}"
exec 9>"${root}/runs/run_currentgf_training_proposals_707_20260823.lock"
flock -n 9 || { echo "current-GF 707 proposal producer already active" >&2; exit 1; }
[[ -s "${marker}" ]] && exit 0
for required in "${manifest}" "${pilot_manifest}" "${checkpoint}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${pilot}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done

"${python_bin}" - "${pilot_manifest}" "${manifest}" "${pilot}" "${output}" <<'PY'
import json
import os
import sys
from pathlib import Path

pilot_manifest, manifest, pilot, output = map(Path, sys.argv[1:])
pilot_names = {str(item["prediction_name"]) for item in json.loads(pilot_manifest.read_text())["items"]}
all_names = {str(item["prediction_name"]) for item in json.loads(manifest.read_text())["items"]}
if len(pilot_names) != 210 or len(all_names) != 707 or not pilot_names < all_names:
    raise SystemExit(f"manifest relation mismatch pilot={len(pilot_names)} all={len(all_names)}")
for name in sorted(pilot_names):
    source = pilot / name
    target = output / name
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        if not target.is_file() or not os.path.samefile(source, target):
            raise SystemExit(f"pilot reuse collision: {target}")
    else:
        os.link(source, target)
PY

# Reuse/audit the pilot immediately; only new GPU inference waits behind the
# bounded weak-transfer gate.
while [[ ! -s "${weak_transfer}" || ! -s "${epoch105_complete}" ]]; do
  printf '%s reused 210 pilot cases; waiting for weak-transfer and epoch105 priority proposal gates\n' "$(date -Ins)"
  sleep 30
done

coverage_complete=0
if "${python_bin}" - "${manifest}" "${output}" <<'PY'
import json,sys
from pathlib import Path
manifest,output=map(Path,sys.argv[1:])
expected={str(item["prediction_name"]) for item in json.loads(manifest.read_text())["items"]}
observed={path.name for path in output.glob("*.nii.gz")}
raise SystemExit(0 if len(expected)==707 and observed==expected else 1)
PY
then
  coverage_complete=1
  printf '%s reusing exact-707 manifest-matched proposal cache; skipping GPU scan\n' "$(date -Ins)"
fi

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
free0=0
free1=0
while [[ "${coverage_complete}" -eq 0 ]]; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

run_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold 0.50 \
    --focus_batch_size 16 --workers 0 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${output}/shard${shard}.log" 2>&1
}
if [[ "${coverage_complete}" -eq 0 && "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  run_shard 0 0 & p0=$!; run_shard 1 1 & p1=$!
  status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${coverage_complete}" -eq 0 ]]; then
  # Keep the exact batch-16 inference recipe serial when only one device is
  # available. A later whole-body case drove two concurrent processes above
  # 49 GiB even though their early measured allocation was lower.
  gpu=1; [[ "${free0}" -eq 1 ]] && gpu=0
  run_shard "${gpu}" 0
  run_shard "${gpu}" 1
fi

"${python_bin}" - "${manifest}" "${output}" "${checkpoint}" "${marker}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

manifest, output, checkpoint, marker = map(Path, sys.argv[1:])
items = json.loads(manifest.read_text(encoding="utf-8"))["items"]
expected = {str(item["prediction_name"]) for item in items}
observed = {path.name for path in output.glob("*.nii.gz")}
if len(items) != 707 or len(expected) != 707 or observed != expected:
    raise SystemExit(
        f"coverage mismatch items={len(items)} names={len(expected)} observed={len(observed)} "
        f"missing={sorted(expected-observed)[:3]} extra={sorted(observed-expected)[:3]}"
    )
hasher = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        hasher.update(block)
digest = hasher.hexdigest()
payload = {
    "schema_version": 1,
    "created_at": datetime.now().astimezone().isoformat(),
    "cases": 707,
    "reused_pilot_cases": 210,
    "generated_additional_cases": 497,
    "labels_read": False,
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": digest,
    "checkpoint_identity": "alpha0.55 epoch105-to-275 initialization; bbox partial-label best model saved at epoch 280",
    "output": str(output.resolve()),
    "window": [-175, 250],
    "glance_threshold": 0.50,
    "lesion_threshold": 0.50,
    "focus_batch_size": 16,
    "workers": 0,
    "component_filter_mm3": 0,
    "public_or_hidden_data_read": False,
    "submission_created": False,
}
temporary = marker.with_name(f".{marker.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(marker)
print(json.dumps(payload, indent=2))
PY
