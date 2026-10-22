#!/usr/bin/env bash
set -euo pipefail

# Generate label-blind high-recall epoch-105 proposals for the 210-case
# complete-label pilot cohort. Labels are not read in this stage.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
images="${root}/analysis/fpr_training_mining_20260822/staged_images"
output="${root}/analysis/candidate_refiner_v2_training_20260822/epoch105_abd_g090_l050"
marker="${output}/PROPOSALS_COMPLETE.json"
expected=210

[[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
[[ -d "${images}" ]] || { echo "missing image directory: ${images}" >&2; exit 2; }
mkdir -p "${output}"
exec 9>"${root}/runs/run_epoch105_training_proposals_210_20260822.lock"
flock -n 9 || { echo "epoch-105 proposal generation already active" >&2; exit 1; }

count=$(find "${output}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
if [[ "${count}" -ne "${expected}" ]]; then
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
      --test_data_path "${images}" --save_prediction_path "${output}" \
      --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
      --glance_threshold 0.90 --lesion_threshold 0.50 \
      --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
      --skip_existing > "${output}/shard${shard}.log" 2>&1 &
    eval "pid${shard}=\$!"
  done
  status=0
  wait "${pid0}" || status=$?
  wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
fi

count=$(find "${output}" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
[[ "${count}" -eq "${expected}" ]] || {
  echo "expected ${expected} predictions, found ${count}" >&2
  exit 2
}
"${python_bin}" - "${checkpoint}" "${output}" "${marker}" "${count}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path

checkpoint = Path(sys.argv[1])
output = Path(sys.argv[2])
marker = Path(sys.argv[3])
count = sys.argv[4]
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(block)
payload = {
    "created_at": datetime.now().astimezone().isoformat(),
    "cases": int(count),
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": digest.hexdigest(),
    "output": str(output.resolve()),
    "window": [-175, 250],
    "glance_threshold": 0.90,
    "lesion_threshold": 0.50,
    "labels_read": False,
}
marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
cat "${marker}"
